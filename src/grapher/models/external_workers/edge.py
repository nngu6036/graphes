"""EDGE adapter for the supplied archive, whose datasets/ package is absent."""
import argparse
import math

import numpy as np
import torch

from common import read_job, load_split, torch_load, save_checkpoint, finish, export_samples, finite_loss


def data_record(adjacency, degree=None):
    from torch_geometric.data import Data
    a = torch.as_tensor(adjacency, dtype=torch.long)
    n = len(a)
    if n < 2:
        raise ValueError("EDGE's edge likelihood requires at least two vertices; no graphs are silently dropped.")
    pairs = torch.triu_indices(n, n, offset=1)
    full = a[pairs[0], pairs[1]]
    present = pairs[:, full.bool()]
    return Data(num_nodes=n, node_attr=torch.zeros(n, dtype=torch.long),
                full_edge_index=pairs, full_edge_attr=full,
                edge_index=present, edge_attr=full[full.bool()],
                degree=a.sum(-1) if degree is None else torch.as_tensor(degree, dtype=torch.long),
                nodes_per_graph=torch.tensor([n]), edges_per_graph=torch.tensor([pairs.size(1)]))


class EmpiricalDegreeSampler:
    def __init__(self, data):
        self.records = [np.sum(a[:n, :n] != 0, axis=-1) for a, n in zip(data["adjacency"], data["num_nodes"])]

    def sample(self, count):
        from torch_geometric.data import Batch
        records = []
        for i in np.random.choice(len(self.records), count, replace=True):
            # Sample the whole indexed degree vector from TRAIN, then permute.
            # Only this invariant is reused; the graph starts empty.
            degree = np.random.permutation(self.records[i])
            records.append(data_record(np.zeros((len(degree), len(degree)), dtype=np.int64), degree))
        return Batch.from_data_list(records)


def make_model(job, train, device, saved=None):
    from model import add_model_args, get_model
    parser = argparse.ArgumentParser(add_help=False)
    add_model_args(parser)
    args = parser.parse_args([])
    updates = dict(job["options"].get("model", {})) if saved is None else dict(saved)
    allowed = set(vars(args)) | {"degree", "has_node_feature", "max_degree", "num_node_classes", "num_edge_classes",
                                "num_node_feat", "augmented_feature_dict", "device"}
    if set(updates) - allowed:
        raise ValueError("Unknown EDGE model options: " + str(set(updates) - allowed))
    vars(args).update(updates)
    args.device = str(device)
    args.num_node_classes = args.num_edge_classes = 2
    args.num_node_feat = 1
    args.has_node_feature = False
    args.degree = True
    args.augmented_feature_dict = {}
    args.max_degree = max(int((a[:n, :n] != 0).sum(-1).max()) for a, n in zip(train["adjacency"], train["num_nodes"]))
    args.max_degree = max(1, args.max_degree)
    args.final_prob_node = [1 - 1e-12, 1e-12]
    args.final_prob_edge = [1 - 1e-12, 1e-12]
    if args.arch != "TGNN_degree_guided" or args.parametrization != "xt_prescribed_st":
        raise ValueError("EDGE adapter requires the supplied degree-guided active-node model.")
    if args.noise_schedule == "linear" and args.diffusion_steps <= 20:
        raise ValueError("Native EDGE linear schedule requires diffusion_steps > 20; use cosine for a tiny smoke test.")
    return get_model(args, EmpiricalDegreeSampler(train)).to(device), args


def main():
    job, device = read_job()
    train = load_split(job)
    if job["stage"] == "train":
        from torch_geometric.loader import DataLoader
        model, args = make_model(job, train, device)
        cfg = job["options"]["train"]
        epochs, batch_size = int(cfg["epochs"]), int(cfg["batch_size"])
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-4)))
        loaders = {}
        for split in ("train", "val"):
            data = train if split == "train" else load_split(job, "val")
            records = [data_record(a[:n, :n]) for a, n in zip(data["adjacency"], data["num_nodes"])]
            loaders[split] = DataLoader(records, batch_size=batch_size, shuffle=split == "train")
        history = []
        for epoch in range(epochs):
            values = {}
            for split, loader in loaders.items():
                model.train(split == "train")
                total, count = 0., 0
                with torch.set_grad_enabled(split == "train"):
                    for batch in loader:
                        batch = batch.to(device)
                        if split == "train": optimizer.zero_grad(set_to_none=True)
                        log_prob = model.log_prob(batch)
                        # Native diffusion.loss.elbo_bpd algebra, without its
                        # unused import of the missing datasets.data_utils.
                        loss = -log_prob.sum() / (math.log(2) * batch.num_entries)
                        finite_loss(loss, "EDGE ELBO")
                        if split == "train":
                            loss.backward()
                            torch.nn.utils.clip_grad_value_(model.parameters(), float(cfg.get("clip_value", 1.)))
                            optimizer.step()
                        total += loss.item() * batch.num_graphs
                        count += batch.num_graphs
                values[split] = total / count
            history.append({"epoch": epoch + 1, **values})
            if (epoch + 1) % int(cfg.get("log_every", 1)) == 0 or epoch + 1 == epochs:
                print("EDGE epoch %d/%d train_bpd=%.6f val_bpd=%.6f" % (epoch + 1, epochs, values["train"], values["val"]), flush=True)
        save_checkpoint(job["checkpoint"], {"model": model.state_dict(), "args": vars(args),
                        "optimizer": optimizer.state_dict(), "epoch": epochs, "history": history})
        finish(job, {"epochs": epochs, "degree_sampler": "empirical_train_only", "attributed": False})
    else:
        state = torch_load(job["checkpoint"], device)
        model, _ = make_model(job, train, device, state["args"])
        model.load_state_dict(state["model"])
        model.eval()
        output, sizes = [], []
        nmax = job["profile"]["max_nodes"]
        bs = int(job["options"].get("generation_batch_size", 32))
        with torch.no_grad():
            for start in range(0, job["num_graphs"], bs):
                b = min(bs, job["num_graphs"] - start)
                batch = model.sample(b)
                # Read full-edge states directly. Native sample() updates only
                # edge_index's PyG slice metadata; to_data_list is unnecessary.
                ptr = batch.ptr.cpu().numpy()
                pairs = batch.full_edge_index.cpu().numpy()
                labels = batch.log_full_edge_attr_t.argmax(-1).cpu().numpy()
                for i in range(b):
                    lo, hi = int(ptr[i]), int(ptr[i + 1])
                    n = hi - lo
                    a = np.zeros((nmax, nmax), dtype=np.int8)
                    select = (pairs[0] >= lo) & (pairs[0] < hi) & (labels != 0)
                    u, v = pairs[:, select] - lo
                    a[u, v] = 1; a[v, u] = 1
                    output.append(a); sizes.append(n)
                print("EDGE generated %d/%d" % (len(sizes), job["num_graphs"]), flush=True)
        export_samples(job, np.stack(output), sizes)


if __name__ == "__main__":
    main()
