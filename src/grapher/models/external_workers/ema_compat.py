"""Complete EMA bookkeeping fallback for an isolated SPECTRE worker.

The update-count-adjusted recurrence is the one in the supplied CatFlow ema.py
and used by torch_ema. This is not a neural-model substitute. It is used only
when torch_ema is absent; model weights, losses and GAN updates remain native.
"""
from contextlib import contextmanager
import weakref
import torch


class ExponentialMovingAverage:
    def __init__(self, parameters, decay, use_num_updates=True):
        if not 0 <= decay <= 1:
            raise ValueError("EMA decay must be in [0, 1].")
        parameters = list(parameters)
        self.decay = float(decay)
        self.num_updates = 0 if use_num_updates else None
        self._refs = [weakref.ref(p) for p in parameters]
        self.shadow_params = [p.detach().clone() for p in parameters]
        self.collected_params = None

    def _parameters(self, parameters):
        parameters = [ref() for ref in self._refs] if parameters is None else list(parameters)
        if len(parameters) != len(self.shadow_params) or any(p is None for p in parameters):
            raise ValueError("EMA parameter count changed or parameters were freed.")
        return parameters

    @torch.no_grad()
    def update(self, parameters=None):
        parameters = self._parameters(parameters)
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        for shadow, param in zip(self.shadow_params, parameters):
            shadow.sub_((1 - decay) * (shadow - param.detach()))

    @torch.no_grad()
    def copy_to(self, parameters=None):
        for shadow, param in zip(self.shadow_params, self._parameters(parameters)):
            param.copy_(shadow)

    def store(self, parameters=None):
        self.collected_params = [p.detach().clone() for p in self._parameters(parameters)]

    @torch.no_grad()
    def restore(self, parameters=None):
        if self.collected_params is None:
            raise RuntimeError("EMA restore requires a preceding store.")
        for saved, param in zip(self.collected_params, self._parameters(parameters)):
            param.copy_(saved)
        self.collected_params = None

    @contextmanager
    def average_parameters(self, parameters=None):
        parameters = self._parameters(parameters)
        self.store(parameters)
        self.copy_to(parameters)
        try:
            yield
        finally:
            self.restore(parameters)

    def to(self, device=None, dtype=None):
        self.shadow_params = [v.to(device=device, dtype=dtype) for v in self.shadow_params]
        if self.collected_params is not None:
            self.collected_params = [v.to(device=device, dtype=dtype) for v in self.collected_params]
        return self

    def state_dict(self):
        return {"decay": self.decay, "num_updates": self.num_updates,
                "shadow_params": self.shadow_params, "collected_params": self.collected_params}

    def load_state_dict(self, state):
        if len(state["shadow_params"]) != len(self.shadow_params):
            raise ValueError("EMA checkpoint has the wrong parameter count.")
        self.decay = float(state["decay"])
        self.num_updates = state["num_updates"]
        self.shadow_params = [v.detach().clone() for v in state["shadow_params"]]
        self.collected_params = state.get("collected_params")
