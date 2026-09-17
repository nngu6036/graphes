"""Device-alias regressions; mocked CUDA cases do not require a GPU."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from grapher.models.dhvae_hh.degree_sampler import DegreeVAESampler


class _DeviceOnlyModel:
    """Expose placement metadata without allocating a CUDA tensor."""

    def __init__(self, device):
        self.parameter = SimpleNamespace(device=torch.device(device))

    def parameters(self):
        return iter([self.parameter])


def _mock_cuda(monkeypatch, current=0, available=True):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)


@pytest.mark.parametrize("requested,current,actual", [
    ("gpu", 0, "cuda:0"),
    ("cuda", 0, "cuda:0"),
    ("auto", 0, "cuda:0"),
    ("gpu", 2, "cuda:2"),
    ("cuda", 2, "cuda:2"),
    ("auto", 2, "cuda:2"),
    (torch.device("cuda"), 1, "cuda:1"),
    ("cuda:0", 2, "cuda:0"),
    ("cuda:2", 0, "cuda:2"),
    ("cpu", 2, "cpu"),
    ("cpu:0", 2, "cpu"),
])
def test_embedded_sampler_pins_alias_and_reuses_model(monkeypatch, requested, current, actual):
    _mock_cuda(monkeypatch, current)
    model, vectorizer = _DeviceOnlyModel(actual), object()
    sampler = DegreeVAESampler.from_config(
        {"checkpoint_path": "unused.pt", "device": requested},
        model=model, vectorizer=vectorizer,
    )
    assert sampler.device == torch.device(actual)
    assert sampler._model is model
    assert sampler._vectorizer is vectorizer
    assert model.parameter.device == torch.device(actual)


@pytest.mark.parametrize("requested,current,actual", [
    ("cuda:1", 0, "cuda:0"),
    ("cuda", 1, "cuda:0"),
    ("gpu", 0, "cuda:2"),
    ("cpu", 0, "cuda:0"),
    ("cuda:0", 0, "cpu"),
])
def test_embedded_sampler_rejects_real_mismatch_with_devices(monkeypatch, requested, current, actual):
    _mock_cuda(monkeypatch, current)
    model = _DeviceOnlyModel(actual)
    with pytest.raises(ValueError, match="Embedded degree model") as error:
        DegreeVAESampler("unused.pt", device=requested, model=model, vectorizer=object())
    assert f"model={actual}" in str(error.value)
    assert "sampler=" in str(error.value)
    assert model.parameter.device == torch.device(actual)


def test_cpu_auto_does_not_query_cuda_current_device(monkeypatch):
    _mock_cuda(monkeypatch, available=False)
    def unexpected():
        raise AssertionError("CPU sampling must not initialise CUDA")
    monkeypatch.setattr(torch.cuda, "current_device", unexpected)
    model = torch.nn.Linear(2, 2)
    sampler = DegreeVAESampler("unused.pt", device="auto", model=model, vectorizer=object())
    assert sampler.device == torch.device("cpu")
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        DegreeVAESampler("unused.pt", device="cuda", model=model, vectorizer=object())


def test_nonembedded_loader_receives_pinned_cuda_device(monkeypatch):
    import grapher.models.dhvae_hh.degree_sampler as module
    _mock_cuda(monkeypatch, current=2)
    calls = []
    model, vectorizer = _DeviceOnlyModel("cuda:2"), object()
    def load(path, *, device):
        calls.append(device)
        return model, vectorizer, {}
    monkeypatch.setattr(module, "load_degree_vae_checkpoint", load)
    sampler = DegreeVAESampler("unused.pt", device="gpu")
    assert calls == [torch.device("cuda:2")]
    assert sampler.device == torch.device("cuda:2")
    assert sampler._model is model


def test_sampling_uses_pinned_device_after_current_gpu_changes(monkeypatch):
    _mock_cuda(monkeypatch, current=1)
    model = _DeviceOnlyModel("cuda:1")
    calls = []
    def sample_outputs(*args, **kwargs):
        calls.append(kwargs["device"])
        return {}
    model.sample_outputs = sample_outputs
    vectorizer = SimpleNamespace(
        sample_empirical_node_count=lambda rng: 3,
        outputs_to_summaries=lambda outputs, **kwargs: [{"degree_sequence": [2, 2, 2]}],
    )
    sampler = DegreeVAESampler("unused.pt", device="cuda", model=model, vectorizer=vectorizer)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert sampler.sample(np.random.default_rng(0))["degree_sequence"] == [2, 2, 2]
    assert calls == [torch.device("cuda:1")]


@pytest.mark.parametrize("index", [0, 2])
def test_structure3_handoff_passes_loaded_models_concrete_device(monkeypatch, tmp_path, index):
    import grapher.models.dhvae_hh.degree_vae as vae_module
    from grapher.models.gdsm_simple.structured_pipeline import _load_degree_sampler
    _mock_cuda(monkeypatch, current=index)
    path = tmp_path / "degree.pt"
    path.write_bytes(b"mock checkpoint; loader is replaced for this device-only test")
    model = _DeviceOnlyModel(f"cuda:{index}")
    vectorizer = SimpleNamespace(empirical_degree_sequences=[[2, 2, 2]])
    monkeypatch.setattr(vae_module, "load_degree_vae_checkpoint", lambda *a, **kw: (model, vectorizer, {}))
    configs = []
    original = DegreeVAESampler.from_config.__func__
    def recording_factory(cls, data, **kwargs):
        configs.append(dict(data))
        return original(cls, data, **kwargs)
    monkeypatch.setattr(DegreeVAESampler, "from_config", classmethod(recording_factory))
    init = {"degree_generator": {"type": "dhvae", "checkpoint_path": str(path),
                                  "fallback": "error", "postprocess_policy": "reject_only"}}
    sampler, provenance = _load_degree_sampler(
        init, {"basis_degree_sequences": [[2, 2, 2]]}, {}, torch.device("cuda"), 42,
    )
    assert configs[0]["device"] == f"cuda:{index}"
    assert sampler.device == torch.device(f"cuda:{index}")
    assert sampler._model is model
    assert provenance["learned"] and provenance["training_degree_multiset_verified"]
    assert init["degree_generator"].get("device") is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA GPU")
@pytest.mark.parametrize("alias", ["gpu", "cuda", "auto"])
def test_real_cuda_embedded_model(alias):
    concrete = torch.device("cuda", torch.cuda.current_device())
    model = torch.nn.Linear(2, 2).to(concrete)
    sampler = DegreeVAESampler("unused.pt", device=alias, model=model, vectorizer=object())
    assert sampler.device == next(model.parameters()).device == concrete
    with pytest.raises(ValueError, match="Embedded degree model"):
        DegreeVAESampler("unused.pt", device="cpu", model=model, vectorizer=object())
