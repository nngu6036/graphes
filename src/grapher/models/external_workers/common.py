"""Worker-only helpers, compatible with Python 3.8+; no GraphER imports."""
import ast
import hashlib
import json
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch


def read_job():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: worker.py /absolute/path/job.json")
    job = json.loads(Path(sys.argv[1]).read_text())
    sys.path.insert(0, job["source_root"])
    random.seed(job["seed"])
    np.random.seed(job["seed"])
    torch.manual_seed(job["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(job["seed"])
    torch.set_num_threads(int(job["options"].get("runtime", {}).get("cpu_threads", 1)))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    requested = job["options"].get("runtime", {}).get("device", "auto")
    require_cuda = requested in ("gpu", "cuda") or str(requested).startswith("cuda:")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in the selected upstream Python environment.")
    device = torch.device("cuda:0" if torch.cuda.is_available() and requested != "cpu" else "cpu")
    job["_resolved_device"] = str(device)
    return job, device


def load_split(job, split="train"):
    if split not in ("train", "val"):
        raise ValueError("Workers may read only train and validation splits.")
    with np.load(Path(job["dataset_dir"]) / (split + ".npz"), allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def torch_load(path, device):
    # Managed checkpoints contain model hyperparameters as well as tensors.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch releases before weights_only existed.
        return torch.load(path, map_location=device)


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def finish(job, extra=None):
    record = {"format": "grapher_external_worker_v1", "model": job["model"],
              "stage": job["stage"], "seed": job["seed"],
              "torch_version": torch.__version__, "python_version": sys.version,
              "cuda_available": torch.cuda.is_available(), "device": job.get("_resolved_device"),
              "numpy_version": np.__version__,
              "adapter_files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted(Path(__file__).parent.glob("*.py"))}}
    record.update(extra or {})
    Path(job["worker_manifest"]).write_text(json.dumps(record, indent=2) + "\n")


def export_samples(job, adjacency, sizes, node_types=None):
    sizes = np.asarray(sizes)
    if sizes.dtype.kind not in "iu":
        raise TypeError("Sampled graph sizes must be integers.")
    sizes = sizes.astype(np.int64)
    adjacency = np.asarray(adjacency)
    if not np.isfinite(adjacency).all():
        raise FloatingPointError("Non-finite sampled adjacency; no samples were published.")
    if len(sizes) != job["num_graphs"]:
        raise RuntimeError("Sampler did not return the exact requested sample count.")
    width = job["profile"]["max_nodes"]
    if np.any(sizes < 1) or np.any(sizes > width):
        raise ValueError("Sampled sizes are outside the declared node support.")
    if adjacency.dtype.kind not in "biu":
        raise TypeError("Worker must discretize adjacency explicitly before export.")
    if not np.isin(adjacency, [0] + (list(job["profile"]["bond_types"]) or [1])).all():
        raise ValueError("Sampled edge category is outside the declared support.")
    if adjacency.shape != (len(sizes), width, width):
        raise ValueError("Sampler output has the wrong padded shape.")
    if node_types is None:
        node_types = np.zeros((len(sizes), width), dtype=np.int16)
    node_types = np.asarray(node_types)
    if node_types.shape != (len(sizes), width) or node_types.dtype.kind not in "iu":
        raise TypeError("Sampled node categories need an integer B-by-N array.")
    if not np.isfinite(node_types).all():
        raise FloatingPointError("Non-finite sampled node category.")
    adjacency = adjacency.astype(np.int8)
    node_types = node_types.astype(np.int16)
    for i, size in enumerate(sizes):
        np.fill_diagonal(adjacency[i], 0)
        adjacency[i, size:, :] = 0
        adjacency[i, :, size:] = 0
        node_types[i, size:] = -1
    np.savez_compressed(job["output"], adjacency=adjacency, num_nodes=sizes, node_types=node_types)
    finish(job, {"num_generated": len(sizes), "posthoc_repair": False,
                 "largest_component_filter": False, "remove_isolates": False})


def finite_loss(loss, label):
    if not torch.isfinite(loss).all():
        raise FloatingPointError("Non-finite " + label)


def selective_module(name, path, selected=None, globals_dict=None, drop_imports=()):
    """Execute supplied source definitions, not rewritten neural algorithms.

    This avoids eager evaluation-only/dataset imports in the research scripts.
    Selection is explicit and recorded in integration documentation. Missing
    definitions fail loudly; missing learning dependencies are never stubbed.
    """
    path = Path(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    if selected is not None:
        found = {getattr(item, "name", None) for item in tree.body}
        if set(selected) - found:
            raise RuntimeError("Upstream source is missing definitions: " + str(set(selected) - found))
        tree.body = [item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.ClassDef))
                     and item.name in selected]
    else:
        tree.body = [item for item in tree.body
                     if not (isinstance(item, ast.ImportFrom) and item.module in drop_imports)
                     and not (isinstance(item, ast.If) and isinstance(item.test, ast.Compare)
                              and isinstance(item.test.left, ast.Name) and item.test.left.id == "__name__")]
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__dict__.update(globals_dict or {})
    sys.modules[name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module
