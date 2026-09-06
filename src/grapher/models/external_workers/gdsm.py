"""GSDM: native spectral score models, spectral loss and spectral PC sampler."""
import copy
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from common import read_job, load_split, torch_load, save_checkpoint, finish, export_samples, finite_loss


def update(base, changes):
    for k, v in changes.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            update(base[k], v)
        else:
            base[k] = v
    return base


def build_models(config, device):
    from models.ScoreNetwork_X import ScoreNetworkX, ScoreNetworkX_GMH
    from models.ScoreNetwork_A_eigen import ScoreNetworkA_eigen
    m, d = config["model"], config["data"]
    if m["x"] != "ScoreNetworkX" or m["adj"] != "ScoreNetworkA_eigen":
        raise ValueError("This adapter supports the supplied ScoreNetworkX/ScoreNetworkA_eigen spectral configuration.")
    x = ScoreNetworkX(max_feat_num=d["max_feat_num"], depth=m["depth"], nhid=m["nhid"]).to(device)
    keys = ("nhid", "num_layers", "num_linears", "c_init", "c_hid", "c_final", "adim", "num_heads", "conv")
    a = ScoreNetworkA_eigen(max_feat_num=d["max_feat_num"], max_node_num=d["max_node_num"],
                           **{k: m[k] for k in keys}).to(device)
    return x, a


def sdes(config):
    from sde import VPSDE
    result = []
    for key in ("x", "adj"):
        c = config["sde"][key]
        if c["type"] != "VP":
            raise ValueError("The supplied spectral marginal/prior path requires VP SDEs.")
        s = VPSDE(beta_min=c["beta_min"], beta_max=c["beta_max"], N=int(c["num_scales"]))
        s.select_type(config.get("type", "linear"))
        result.append(s)
    return result


def dataset(data, config):
    from utils.graph_utils import init_features
    adj = torch.tensor(data["adjacency"], dtype=torch.float32)
    n = data["num_nodes"]
    for i, size in enumerate(n):
        if np.any(data["adjacency"][i, :size, :size].sum(0) == 0):
            raise ValueError("Native GSDM infers node masks from adjacency; training/validation isolates are unsupported.")
    features = init_features("deg", adj, config["data"]["max_feat_num"])
    la, u = torch.linalg.eigh(adj)
    return TensorDataset(features, adj, u, la)


def main():
    job, device = read_job()
    options = job["options"]
    train = load_split(job)
    if job["stage"] == "train":
        filename = options.get("upstream_config", "community_small.yaml")
        source = Path(filename)
        if not source.is_absolute():
            source = Path(job["source_root"]) / "config" / source
        config = yaml.safe_load(source.read_text())
        update(config, copy.deepcopy(options.get("config_overrides", {})))
        config["type"] = config.get("type", "linear")
        config["data"]["max_node_num"] = job["profile"]["max_nodes"]
        # A fixed maximum node support also bounds degree; no validation/test
        # statistic is used to define the feature vocabulary.
        degree_classes = 5 if job["profile"]["benchmark_id"] == "grid" else job["profile"]["max_nodes"]
        config["data"]["max_feat_num"] = max(config["data"]["max_feat_num"], degree_classes)
        epochs = int(options["train"]["epochs"])
        config["train"].update({k: v for k, v in options["train"].items() if k not in {"epochs", "batch_size", "log_every"}})
        config["train"]["num_epochs"] = epochs
        config["data"]["batch_size"] = int(options["train"]["batch_size"])
        model_x, model_a = build_models(config, device)
        sx, sa = sdes(config)
        from losses import get_sde_loss_fn2
        from utils.ema import ExponentialMovingAverage
        cfg = config["train"]
        loss_fns = {flag: get_sde_loss_fn2(sx, sa, train=flag, reduce_mean=cfg["reduce_mean"], continuous=True,
                     likelihood_weighting=False, eps=cfg["eps"]) for flag in (True, False)}
        models = [model_x, model_a]
        optimizers = [torch.optim.Adam(m.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"])) for m in models]
        schedulers = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=cfg["lr_decay"]) for o in optimizers]
        emas = [ExponentialMovingAverage(m.parameters(), decay=cfg["ema"]) for m in models]
        loaders = {s: DataLoader(dataset(train if s == "train" else load_split(job, "val"), config),
                                batch_size=config["data"]["batch_size"], shuffle=s == "train") for s in ("train", "val")}
        history = []
        for epoch in range(epochs):
            totals = {}
            for split, loader in loaders.items():
                training = split == "train"
                for m in models: m.train(training)
                total = 0.
                with torch.set_grad_enabled(training):
                    for batch in loader:
                        if training:
                            for o in optimizers: o.zero_grad(set_to_none=True)
                        x, a, u, la = (v.to(device) for v in batch)
                        lx, le = loss_fns[training](model_x, model_a, x, a, u, la)
                        loss = lx + le
                        finite_loss(loss, "GSDM spectral loss")
                        if training:
                            loss.backward()
                            for m, o, ema in zip(models, optimizers, emas):
                                torch.nn.utils.clip_grad_norm_(m.parameters(), cfg["grad_norm"])
                                o.step()
                                ema.update(m.parameters())
                        total += loss.item() * x.size(0)
                totals[split] = total / len(loader.dataset)
            if cfg.get("lr_schedule", True):
                for scheduler in schedulers: scheduler.step()
            history.append({"epoch": epoch + 1, **totals})
            if (epoch + 1) % int(options["train"].get("log_every", 1)) == 0 or epoch + 1 == epochs:
                print("GSDM epoch %d/%d train=%.6f val=%.6f" % (epoch + 1, epochs, totals["train"], totals["val"]), flush=True)
        save_checkpoint(job["checkpoint"], {"config": config, "x": model_x.state_dict(), "adj": model_a.state_dict(),
            "ema_x": emas[0].state_dict(), "ema_adj": emas[1].state_dict(), "epoch": epochs,
            "optimizers": [o.state_dict() for o in optimizers], "history": history})
        finish(job, {"epochs": epochs, "config": config, "spectral_basis_source": "train_only"})
    else:
        state = torch_load(job["checkpoint"], device)
        config = state["config"]
        mx, ma = build_models(config, device)
        mx.load_state_dict(state["x"]); ma.load_state_dict(state["adj"])
        sample = update(dict(config["sample"]), options.get("sample", {}))
        if sample.get("use_ema", False):
            from utils.ema import ExponentialMovingAverage
            for model, key in ((mx, "ema_x"), (ma, "ema_adj")):
                ema = ExponentialMovingAverage(model.parameters(), decay=config["train"]["ema"])
                ema.load_state_dict(state[key]); ema.copy_to(model.parameters())
        mx.eval(); ma.eval()
        sx, sa = sdes(config)
        from solver import get_pc_sampler2
        output, sizes = [], []
        bs = int(options.get("generation_batch_size", 128))
        nmax, feats = config["data"]["max_node_num"], config["data"]["max_feat_num"]
        for start in range(0, job["num_graphs"], bs):
            b = min(bs, job["num_graphs"] - start)
            indices = np.random.choice(len(train["num_nodes"]), b, replace=True)
            n = train["num_nodes"][indices]
            flags = (torch.arange(nmax, device=device)[None] < torch.tensor(n, device=device)[:, None]).float()
            bases = torch.tensor(train["adjacency"][indices], dtype=torch.float32, device=device)
            sampler = get_pc_sampler2(sx, sa, (b, nmax, feats), (b, nmax, nmax), **config["sampler"],
                        probability_flow=sample["probability_flow"], continuous=True,
                        denoise=sample["noise_removal"], eps=sample["eps"], device=str(device))
            _, adjacency, _ = sampler(mx, ma, flags, bases)
            if not torch.isfinite(adjacency).all():
                raise FloatingPointError("Non-finite GSDM spectral samples.")
            adjacency = (adjacency + adjacency.transpose(-1, -2)) / 2
            output.append((adjacency > .5).long().cpu().numpy()); sizes.extend(n.tolist())
            print("GSDM generated %d/%d" % (len(sizes), job["num_graphs"]), flush=True)
        export_samples(job, np.concatenate(output), sizes)


if __name__ == "__main__":
    main()
