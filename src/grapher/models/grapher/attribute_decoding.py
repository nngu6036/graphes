"""Topology-preserving categorical decoding for neutral QM9-style molecules.

The optional constraints apply only to neutral C/N/O/F with integer single,
double and triple bonds and implicit hydrogens. They bound weighted valence;
they are not a full chemical-validity test or an exact conditional sampler.
RDKit reference: https://www.rdkit.org/docs/RDKit_Book.html#valence-calculation-and-allowed-valences
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

NEUTRAL_CAPACITIES = {6: 4, 7: 3, 8: 2, 9: 1}
CONSTRAINT_MODES = {"none", "atom_degree", "atom_bond_valence"}


def sample_categorical_logits(
    logits: torch.Tensor, *, mode: str, temperature: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Decode finite logits, allowing -inf for explicitly forbidden categories."""
    mode = str(mode).lower()
    if mode not in {"argmax", "sample", "categorical", "stochastic"}:
        raise ValueError(f"Unsupported categorical decode mode {mode!r}")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("categorical sampling temperature must be finite and > 0")
    if logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError("Expected a nonempty category axis")
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise FloatingPointError("Categorical logits contain NaN or +inf")
    if not torch.isfinite(logits).any(dim=-1).all():
        raise ValueError("Every categorical row must have at least one finite logit")
    if mode == "argmax":
        return logits.argmax(dim=-1)
    # Center before temperature scaling; finite logits can otherwise overflow.
    centered = logits - logits.max(dim=-1, keepdim=True).values
    probs = torch.softmax(centered / float(temperature), dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    if not flat.shape[0]:
        return torch.empty(logits.shape[:-1], dtype=torch.long, device=logits.device)
    return torch.multinomial(flat, 1, replacement=True, generator=generator).reshape(logits.shape[:-1])


def validate_decode_config(
    decode: Mapping[str, Any], node_values: Sequence[Any], edge_values: Sequence[Any],
    *, node_attribute: str | None = "atomic_num", edge_attribute: str | None = "bond_type",
) -> None:
    mode = str(decode.get("constraint_mode", "none"))
    if mode not in CONSTRAINT_MODES:
        raise ValueError(f"constraint_mode must be one of {sorted(CONSTRAINT_MODES)}")
    if str(decode.get("edge_order", "random")) not in {"random", "lexicographic"}:
        raise ValueError("edge_order must be random or lexicographic")
    if str(decode.get("infeasible_policy", "retain")) not in {"retain", "error"}:
        raise ValueError("infeasible_policy must be retain or error (no filtering/resampling)")
    if mode != "none":
        if node_attribute != "atomic_num" or edge_attribute != "bond_type":
            raise ValueError("Neutral valence constraints require atomic_num / bond_type attributes")
        if not node_values or any(v not in NEUTRAL_CAPACITIES for v in node_values):
            raise ValueError("Neutral valence constraints support only C/N/O/F atomic numbers 6,7,8,9")
        if not edge_values or 1 not in edge_values or any(v not in (1, 2, 3) for v in edge_values):
            raise ValueError("Neutral valence constraints require integer bonds in {1,2,3}, including single bonds")


def _capacity_tensor(values: Sequence[Any], device: torch.device) -> torch.Tensor:
    return torch.tensor([NEUTRAL_CAPACITIES[v] for v in values], device=device, dtype=torch.long)


def sample_atoms(
    logits: torch.Tensor, discrete: torch.Tensor, flags: torch.Tensor,
    node_values: Sequence[Any], decode: Mapping[str, Any], generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask insufficient atom capacities; retain infeasible topologies explicitly."""
    constraint = str(decode.get("constraint_mode", "none"))
    applied = torch.zeros_like(flags, dtype=torch.bool)
    active = flags.bool()
    if constraint != "none":
        degree = discrete.long().sum(-1)
        capacities = _capacity_tensor(node_values, logits.device)
        allowed = capacities.view(1, 1, -1) >= degree.unsqueeze(-1)
        feasible = allowed.any(-1)
        impossible = active & ~feasible
        if impossible.any() and str(decode.get("infeasible_policy", "retain")) == "error":
            raise ValueError("Topology has a degree exceeding every supported neutral atom capacity")
        # An impossible node is deliberately left unconstrained and reported as
        # a failed draw. Never remove edges, omit the graph, or replace its topology.
        apply = active & feasible
        applied = apply & ~allowed.all(-1)
        logits = logits.masked_fill(apply.unsqueeze(-1) & ~allowed, -torch.inf)
    result = sample_categorical_logits(
        logits, mode=str(decode.get("node_mode", "argmax")),
        temperature=float(decode.get("node_temperature", 1.0)), generator=generator,
    )
    return result.masked_fill(~active, 0), applied


def sample_bonds(
    logits: torch.Tensor, discrete: torch.Tensor, flags: torch.Tensor,
    node_indices: torch.Tensor, node_values: Sequence[Any], edge_values: Sequence[Any],
    decode: Mapping[str, Any], generator: torch.Generator,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Sample each unordered present edge once, optionally reserving valence.

    Single-bond capacity is reserved for ALL topology edges before allocation.
    Only the additional b-1 units of a double/triple bond consume slack. The
    invariant follows inductively on feasible topologies. On infeasible nodes,
    only single bonds are allocated and the impossible draw remains in outputs.
    """
    constrained = str(decode.get("constraint_mode", "none")) == "atom_bond_valence"
    result = torch.zeros_like(discrete, dtype=torch.long)
    rows: list[dict[str, Any]] = [
        {"bond_constraint_activations": 0, "bond_allocation_order": []}
        for _ in range(discrete.shape[0])
    ]
    upper = torch.triu(discrete.bool(), diagonal=1)
    mode = str(decode.get("edge_mode", "argmax"))
    temperature = float(decode.get("edge_temperature", 1.0))
    if not constrained:
        pos = upper.nonzero(as_tuple=False)
        if pos.numel():
            sampled = sample_categorical_logits(
                logits[pos[:, 0], pos[:, 1], pos[:, 2]], mode=mode,
                temperature=temperature, generator=generator,
            )
            result[pos[:, 0], pos[:, 1], pos[:, 2]] = sampled
            result[pos[:, 0], pos[:, 2], pos[:, 1]] = sampled
        return result, rows

    capacity = _capacity_tensor(node_values, logits.device)[node_indices]
    degree = discrete.long().sum(-1)
    slack = capacity - degree
    bond_orders = torch.tensor(list(edge_values), device=logits.device, dtype=torch.long)
    extras = bond_orders - 1
    for r in range(discrete.shape[0]):
        edges = upper[r].nonzero(as_tuple=False)
        if str(decode.get("edge_order", "random")) == "random" and edges.shape[0]:
            order = torch.randperm(edges.shape[0], device=logits.device, generator=generator)
            edges = edges[order]
        for i, j in edges.tolist():
            # Negative slack is possible only on a retained infeasible draw.
            # A single bond remains the explicit fallback, not a hidden repair.
            budget = torch.minimum(slack[r, i], slack[r, j]).clamp_min(0)
            allowed = extras <= budget
            rows[r]["bond_constraint_activations"] += int(not bool(allowed.all()))
            rows[r]["bond_allocation_order"].append([i, j])
            label = sample_categorical_logits(
                logits[r, i, j].masked_fill(~allowed, -torch.inf), mode=mode,
                temperature=temperature, generator=generator,
            )
            result[r, i, j] = result[r, j, i] = label
            slack[r, i] -= extras[label]
            slack[r, j] -= extras[label]
    return result, rows


def decoding_diagnostics(
    discrete: torch.Tensor, flags: torch.Tensor, node_indices: torch.Tensor,
    edge_indices: torch.Tensor, node_values: Sequence[Any], edge_values: Sequence[Any],
    atom_masks: torch.Tensor, rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Overlapping failure flags, not an assertion of full chemical validity."""
    applicable = (all(v in NEUTRAL_CAPACITIES for v in node_values)
                  and bool(node_values) and all(v in (1, 2, 3) for v in edge_values))
    degree = discrete.long().sum(-1)
    if applicable:
        capacities = _capacity_tensor(node_values, discrete.device)
        selected_capacity = capacities[node_indices]
        orders = torch.tensor(list(edge_values), device=discrete.device)[edge_indices]
        weighted = (orders * discrete).sum(-1)
    for r, row in enumerate(rows):
        active = flags[r].bool()
        row["neutral_valence_diagnostics_applicable"] = bool(applicable)
        row["atom_constraint_activations"] = int(atom_masks[r, active].sum())
        row["topology_preserved"] = True
        if applicable:
            impossible = degree[r, active] > capacities.max()
            atom_bad = degree[r, active] > selected_capacity[r, active]
            valence_bad = weighted[r, active] > selected_capacity[r, active]
            row.update({
                "topology_infeasible": bool(impossible.any()),
                "topology_infeasible_node_count": int(impossible.sum()),
                "atom_degree_incompatible": bool(atom_bad.any()),
                "atom_degree_incompatible_node_count": int(atom_bad.sum()),
                "bond_valence_exceeded": bool(valence_bad.any()),
                "bond_valence_exceeded_node_count": int(valence_bad.sum()),
                "capacity_constraints_satisfied": not bool(valence_bad.any()),
            })
    return rows
