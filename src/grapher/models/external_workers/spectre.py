"""Native SPECTRE GAN updates with a version-independent managed training loop.

The original _disc_step/_gen_step/configure_optimizers and all neural modules
are executed unchanged. Trainer logging/epoch bookkeeping is adapted, avoiding
obsolete Lightning Trainer CLI and evaluation-only graph_tool imports.
"""
import argparse
import inspect
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common import read_job, load_split, torch_load, save_checkpoint, finish, export_samples, finite_loss, selective_module


def runtime_class(job):
    root = Path(job["source_root"])
    # Data is referenced only by native evaluation helpers, never by the dense
    # neural path. Do not import an unused PyG data layer into this worker.
    selective_module("util.model_helper", root / "util" / "model_helper.py",
                     drop_imports=("torch_geometric.data",))
    try:
        from torch_ema import ExponentialMovingAverage
        ema_backend = "torch_ema"
    except ModuleNotFoundError as exc:
        if exc.name != "torch_ema":
            raise
        from ema_compat import ExponentialMovingAverage
        ema_backend = "compat_update_count_ema"
    module = selective_module("full_gan", root / "full_gan.py",
        drop_imports=("data", "util.eval_helper", "torch_ema"),
        globals_dict={"N_MAX": job["profile"]["max_nodes"], "ExponentialMovingAverage": ExponentialMovingAverage})
    class ManagedSPECTRE(module.SPECTRE):
        _epoch = 0
        @property
        def current_epoch(self):
            return self._epoch
        def optimizers(self, use_pl_optimizer=True):
            return self._managed_optimizers
        def log(self, name, value, *args, **kwargs):
            if isinstance(value, torch.Tensor):
                finite_loss(value.detach(), "SPECTRE " + name)
                value = float(value.detach())
            self._metrics[name] = value
        def manual_backward(self, loss, *args, **kwargs):
            finite_loss(loss, "SPECTRE gradient objective")
            loss.backward(*args, **kwargs)
    ManagedSPECTRE.ema_backend = ema_backend
    return ManagedSPECTRE


class SpectralDataset(Dataset):
    def __init__(self, data, nmax, ignore_first, molecular):
        self.data = data
        self.nmax = nmax
        self.ignore_first = ignore_first
        self.molecular = molecular
        self.spectra = []
        for a, n in zip(data["adjacency"], data["num_nodes"]):
            adjacency = torch.tensor(a[:n, :n] != 0, dtype=torch.float32)
            degree = adjacency.sum(-1)
            inv = degree.clamp(min=1).rsqrt()
            laplacian = torch.diag((degree > 0).float()) - inv[:, None] * adjacency * inv[None, :]
            vals, vecs = torch.linalg.eigh(laplacian)
            if ignore_first:
                vals, vecs = vals[1:], vecs[:, 1:]
            vals = F.pad(vals, (0, nmax - len(vals)))
            vecs = F.pad(vecs, (0, nmax - vecs.size(1), 0, nmax - n))
            self.spectra.append((vals, vecs))
    def __len__(self):
        return len(self.data["num_nodes"])
    def __getitem__(self, i):
        n = int(self.data["num_nodes"][i])
        a = torch.tensor(self.data["adjacency"][i]).long()
        mask1 = torch.arange(self.nmax) < n
        mask = (mask1[:, None] & mask1[None, :]).long()
        vals, vecs = self.spectra[i]
        record = {"n_nodes": n, "adj": (a != 0).float(), "eigval": vals, "eigvec": vecs, "mask": mask}
        if self.molecular:
            types = torch.tensor(self.data["node_types"][i]).long().clamp(min=0)
            record["node_features"] = F.one_hot(types, 4).float() * mask1[:, None]
            record["edge_features"] = F.one_hot(a, 4).float()[..., 1:] * mask[..., None]
        return record


def main():
    job, device = read_job()
    Model = runtime_class(job)
    options = job["options"]
    train = load_split(job)
    if job["stage"] == "train":
        defaults = vars(Model.add_model_specific_args(argparse.ArgumentParser(add_help=False)).parse_args([]))
        parameters = inspect.signature(Model.__init__).parameters
        params = {k: v for k, v in defaults.items() if k in parameters}
        overrides = options.get("model", {})
        if set(overrides) - set(parameters):
            raise ValueError("Unknown SPECTRE model options: " + str(set(overrides) - set(parameters)))
        params.update(overrides)
        params["n_max"] = job["profile"]["max_nodes"]
        params["qm9"] = job["profile"]["benchmark_id"] == "qm9"
        for flag in ("adj_only", "adj_eigvec_only", "SON_only", "lambda_only", "lambda_SON_only", "mlp_gen", "no_cond", "use_fixed_emb"):
            if params.get(flag, False):
                raise ValueError("Unconditional SPECTRE wrapper does not enable conditional/ablation flag " + flag)
        if int(params.get("pretrain", 0)):
            raise ValueError("Pretraining-only SPECTRE phases are not supported by this managed adapter.")
        k = int(params["k_eigval"])
        if not 2 <= k <= job["profile"]["max_nodes"]:
            raise ValueError("Native SPECTRE's spectral-normalized rotation heads require 2 <= k_eigval <= max_nodes.")
        singleton_mask = train["num_nodes"] == 1
        singleton_policy = options["train"].get("singleton_policy", "error")
        if singleton_policy not in {"error", "empirical"}:
            raise ValueError("train.singleton_policy must be error or empirical.")
        if singleton_mask.any() and singleton_policy != "empirical":
            raise ValueError("Native SPECTRE cannot process singleton graphs. Set train.singleton_policy=empirical "
                             "to model their atom/category distribution as a separate, recorded size-one branch.")
        singleton_types = train["node_types"][singleton_mask, 0].tolist()
        gan_train = {key: value[~singleton_mask] for key, value in train.items()}
        if not len(gan_train["num_nodes"]):
            raise ValueError("SPECTRE needs at least one nonsingleton training graph.")
        # Like the native dataset loader, pad unavailable trailing eigenmodes
        # with zeros. Never invent extra vertices or discard two-node graphs.
        padded_count = int((gan_train["num_nodes"] - int(params.get("ignore_first_eigv", False)) < k).sum())
        model = Model(**params).to(device)
        model._metrics = {}
        model._managed_optimizers, schedulers = model.configure_optimizers()
        cfg = options["train"]
        epochs = int(cfg["epochs"])
        dataset = SpectralDataset(gan_train, params["n_max"], params.get("ignore_first_eigv", False), params["qm9"])
        loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=True)
        history = []
        for epoch in range(epochs):
            model._epoch = epoch
            model.train()
            totals, count = {}, 0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                width = max(int(batch["n_nodes"].max()), k)
                values = batch["eigval"]
                vectors = batch["eigvec"][:, :width, :k]
                mask = batch["mask"][:, :width, :width]
                adjacency = batch["adj"][:, :width, :width]
                edge = batch["edge_features"][:, :width, :width] if params["qm9"] else None
                node = batch["node_features"][:, :width] if params["qm9"] else None
                model.SON_generator.gumbel_temperature = model._SON_gumbel_temp()
                model._disc_step(values, vectors, mask, adjacency, edge, node)
                model._gen_step(values, vectors, mask)
                for name in ("loss/disc", "loss/gen"):
                    totals[name] = totals.get(name, 0.) + model._metrics[name]
                count += 1
            if params.get("SON_init_bank_size", 0) > 0:
                model.SON_generator.bank_sample_hist.zero_()
            every = int(params.get("lr_decay_every", 10))
            if (epoch + 1) % every == 0 and epoch + 1 >= int(params.get("lr_decay_warmup", 10)):
                for scheduler in schedulers: scheduler.step()
            history.append({"epoch": epoch + 1, **{k: v / count for k, v in totals.items()}})
            if (epoch + 1) % int(cfg.get("log_every", 1)) == 0 or epoch + 1 == epochs:
                print("SPECTRE epoch %d/%d D=%.6f G=%.6f" % (epoch + 1, epochs, totals["loss/disc"] / count, totals["loss/gen"] / count), flush=True)
        save_checkpoint(job["checkpoint"], {"model": model.state_dict(), "params": params, "epoch": epochs,
                "ema": model.gen_ema.state_dict(), "ema_backend": Model.ema_backend, "history": history,
                "singleton_types": singleton_types, "singleton_policy": singleton_policy,
                "optimizers": [o.state_dict() for o in model._managed_optimizers]})
        finish(job, {"epochs": epochs, "ema_backend": Model.ema_backend, "checkpoint_selection": "final_epoch", "validation_used": False,
                     "training_loop": "native_discriminator_and_generator_updates",
                     "gan_training_count": len(gan_train["num_nodes"]), "padded_eigenmode_count": padded_count,
                     "singleton_count": len(singleton_types), "singleton_policy": singleton_policy})
    else:
        state = torch_load(job["checkpoint"], device)
        model = Model(**state["params"]).to(device)
        model._metrics = {}
        model._epoch = state["epoch"]
        model.load_state_dict(state["model"])
        sample = options.get("sample", {})
        if sample.get("use_ema", True):
            model.gen_ema.load_state_dict(state["ema"])
            model.gen_ema.to(device)
            model.gen_ema.copy_to(model.gen_params)
        model.eval()
        nmax, k = state["params"]["n_max"], state["params"]["k_eigval"]
        all_a, all_n, all_x = [], [], []
        bs = int(options.get("generation_batch_size", 32))
        with torch.no_grad():
            for start in range(0, job["num_graphs"], bs):
                b = min(bs, job["num_graphs"] - start)
                sizes = np.random.choice(train["num_nodes"], b, replace=True)
                batch_a = np.zeros((b, nmax, nmax), dtype=np.int8)
                batch_x = np.zeros((b, nmax), dtype=np.int16)
                single = np.flatnonzero(sizes == 1)
                large = np.flatnonzero(sizes > 1)
                if len(single):
                    support = state.get("singleton_types", [])
                    if state.get("singleton_policy") != "empirical" or not support:
                        raise RuntimeError("Singleton size sampled without a fitted singleton branch.")
                    batch_x[single, 0] = np.random.choice(support, len(single), replace=True)
                if len(large):
                    n = sizes[large]
                    width = max(int(n.max()), k)
                    mask1 = torch.arange(width, device=device)[None] < torch.tensor(n, device=device)[:, None]
                    mask = (mask1[:, :, None] & mask1[:, None, :]).float()
                    # all_fake sets both real-eigenvalue/eigenvector mixing
                    # probabilities to zero. No real spectra are supplied.
                    vals = torch.zeros(len(large), k, device=device)
                    vecs = torch.zeros(len(large), width, k, device=device)
                    result = model._get_fake(vals, vecs, mask, test_type="all_fake")
                    adj, nodes, edges = result[:3]
                    if not torch.isfinite(adj).all():
                        raise FloatingPointError("SPECTRE generated non-finite adjacency.")
                    if state["params"]["qm9"]:
                        if not torch.isfinite(nodes).all() or not torch.isfinite(edges).all():
                            raise FloatingPointError("SPECTRE generated non-finite categorical features.")
                        # Preserve full_gan.validation_function's two-step rule:
                        # threshold total edge probability, then choose bond type.
                        categories = (edges.argmax(-1) + 1) * (adj > .5).long()
                        node_ids = nodes.argmax(-1)
                    else:
                        categories = (adj > .5).long()
                        node_ids = torch.zeros(len(large), width, dtype=torch.long, device=device)
                    categories = torch.triu(categories, diagonal=1)
                    categories = categories + categories.transpose(-1, -2)
                    batch_a[large] = F.pad(categories, (0, nmax - width, 0, nmax - width)).cpu().numpy()
                    batch_x[large] = F.pad(node_ids, (0, nmax - width)).cpu().numpy()
                all_a.append(batch_a)
                all_x.append(batch_x)
                all_n.extend(sizes.tolist())
                print("SPECTRE generated %d/%d" % (len(all_n), job["num_graphs"]), flush=True)
        export_samples(job, np.concatenate(all_a), all_n, np.concatenate(all_x))


if __name__ == "__main__":
    main()
