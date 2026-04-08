from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from collections import deque

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Region:
    """Half-open rectangle on patch grid: rows [r0,r1), cols [c0,c1)."""
    r0: int
    r1: int
    c0: int
    c1: int

    @property
    def h(self) -> int:
        return self.r1 - self.r0

    @property
    def w(self) -> int:
        return self.c1 - self.c0

    @property
    def area(self) -> int:
        return self.h * self.w


@dataclass
class Node:
    """A node in the recursive partition tree."""
    node_id: int
    level: int
    region: Region
    parent: Optional[int]
    children: Optional[Tuple[int, ...]]


class QuadTreeBuilder:
    """
    Build a full recursive partition tree down to 1x1 patch leaves.

    Partition rule:
      - h >= 2, w >= 2: 4-way split
      - h >= 2, w == 1: vertical 2-way split
      - h == 1, w >= 2: horizontal 2-way split
      - h == 1, w == 1: leaf
    """

    @staticmethod
    def validate_hw(N: int, H: int, W: int) -> Tuple[int, int]:
        H, W = int(H), int(W)
        if H <= 0 or W <= 0:
            raise ValueError(f"H and W must be positive, got H={H}, W={W}")
        if H * W != N:
            raise ValueError(f"Invalid grid size: H*W={H*W}, but N={N}")
        return H, W

    @staticmethod
    def can_split(reg: Region) -> bool:
        return not (reg.h == 1 and reg.w == 1)

    @staticmethod
    def split_region(reg: Region) -> Optional[Tuple[Region, ...]]:
        h, w = reg.h, reg.w
        if h == 1 and w == 1:
            return None
        if h >= 2 and w >= 2:
            rm = (reg.r0 + reg.r1) // 2
            cm = (reg.c0 + reg.c1) // 2
            if rm == reg.r0 or rm == reg.r1 or cm == reg.c0 or cm == reg.c1:
                return None
            return (
                Region(reg.r0, rm, reg.c0, cm),
                Region(reg.r0, rm, cm, reg.c1),
                Region(rm, reg.r1, reg.c0, cm),
                Region(rm, reg.r1, cm, reg.c1),
            )
        if h >= 2 and w == 1:
            rm = (reg.r0 + reg.r1) // 2
            if rm == reg.r0 or rm == reg.r1:
                return None
            return Region(reg.r0, rm, reg.c0, reg.c1), Region(rm, reg.r1, reg.c0, reg.c1)
        if h == 1 and w >= 2:
            cm = (reg.c0 + reg.c1) // 2
            if cm == reg.c0 or cm == reg.c1:
                return None
            return Region(reg.r0, reg.r1, reg.c0, cm), Region(reg.r0, reg.r1, cm, reg.c1)
        return None

    @torch.no_grad()
    def build(self, x: torch.Tensor, H: int, W: int) -> Dict[str, Any]:
        B, N, D = x.shape
        H, W = self.validate_hw(N, H, W)

        nodes: List[Node] = []
        next_id = 0
        nodes.append(Node(node_id=0, level=0, region=Region(0, H, 0, W), parent=None, children=None))
        next_id += 1

        queue = deque([0])
        while queue:
            pid = queue.popleft()
            preg = nodes[pid].region
            if not self.can_split(preg):
                continue
            sub = self.split_region(preg)
            if sub is None:
                continue
            child_ids = []
            for reg in sub:
                cid = next_id
                next_id += 1
                nodes.append(Node(node_id=cid, level=nodes[pid].level + 1, region=reg, parent=pid, children=None))
                child_ids.append(cid)
            nodes[pid].children = tuple(child_ids)
            queue.extend(child_ids)

        return {"H": H, "W": W, "nodes": nodes}


class QuadTreeNavigator:
    """
    Navigate the tree using a precomputed patch score map.

    For each node:
      1) Prune if softmax-pooled score < global softmax-pooled score
         (softmax pooling is sensitive to local peaks, prevents pruning regions
          with small but important areas)
      2) Split if coefficient of variation (std/mean) > split_threshold
      3) Otherwise: select this node

    Two parameters with decoupled roles:
      softmax_temperature: controls pruning sensitivity to local high-score patches.
                           Fixed at a sensible default, not exposed to users.
      split_threshold:     controls splitting granularity. The only tunable param.
                           Higher → coarser regions. Lower → finer regions.
    """

    def __init__(self, split_threshold: float = 0.5, softmax_temperature: float = 0.3, eps: float = 1e-6):
        self.split_threshold = float(split_threshold)
        self.softmax_temperature = float(softmax_temperature)
        self.eps = float(eps)

    @staticmethod
    def region_to_token_indices(reg: Region, W: int, device: torch.device) -> torch.Tensor:
        rows = torch.arange(reg.r0, reg.r1, device=device)
        cols = torch.arange(reg.c0, reg.c1, device=device)
        rr, cc = torch.meshgrid(rows, cols, indexing="ij")
        return (rr * W + cc).reshape(-1)

    def _softmax_pool(self, vals: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(vals / self.softmax_temperature, dim=0)
        return torch.sum(weights * vals)

    @torch.no_grad()
    def select_nodes(self, nodes: List[Node], patch_scores: torch.Tensor, W: int):
        B, N = patch_scores.shape

        # global mean — pruning baseline
        global_mean = patch_scores.mean(dim=1)  # [B]

        selected = [[] for _ in range(B)]
        visited = [[] for _ in range(B)]

        for b in range(B):
            Q = deque([0])
            while Q:
                pid = Q.popleft()
                visited[b].append(pid)

                reg = nodes[pid].region
                idx = self.region_to_token_indices(reg, W, patch_scores.device)
                vals = patch_scores[b, idx]

                # prune: local softmax pool vs global mean
                # softmax_temperature controls sensitivity:
                #   low temp  → pool dominated by peak → less pruning (sensitive to local highlights)
                #   high temp → pool approaches mean   → more pruning (stricter)
                s_soft = self._softmax_pool(vals)
                if s_soft < global_mean[b]:
                    continue

                children = nodes[pid].children
                if not children:
                    selected[b].append(pid)
                    continue

                # split: coefficient of variation measures internal score spread
                mean = vals.mean()
                std = vals.std()
                cv = std / (mean + self.eps)

                if cv > self.split_threshold:
                    Q.extend(children)
                else:
                    selected[b].append(pid)

        return selected, visited

    @torch.no_grad()
    def nodes_to_tokens(
        self,
        nodes: List[Node],
        H: int,
        W: int,
        selected_node_ids: List[List[int]],
        x: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        device = x.device if x is not None else torch.device("cpu")
        B = len(selected_node_ids)

        token_indices: List[torch.Tensor] = []
        for b in range(B):
            idxs = [self.region_to_token_indices(nodes[nid].region, W=W, device=device)
                    for nid in selected_node_ids[b]]
            token_indices.append(
                torch.cat(idxs, dim=0).unique(sorted=True) if idxs
                else torch.empty(0, device=device, dtype=torch.long)
            )

        out: Dict[str, Any] = {"selected_token_indices": token_indices}

        if x is not None:
            Bx, N, D = x.shape
            assert Bx == B and N == H * W, f"x must be [B,H*W,D], got {x.shape}, H*W={H*W}"
            Mmax = max(int(t.numel()) for t in token_indices) if B > 0 else 0
            feats = torch.zeros((B, Mmax, D), device=device, dtype=x.dtype)
            mask = torch.zeros((B, Mmax), device=device, dtype=torch.bool)
            for b in range(B):
                idx = token_indices[b]
                m = int(idx.numel())
                if m > 0:
                    feats[b, :m, :] = x[b, idx, :]
                    mask[b, :m] = True
            out["selected_feats_padded"] = feats
            out["selected_mask"] = mask

        return out



class QVTree(nn.Module):
    """QuadTree-based visual token selector. Uses builder + navigator; scorer is external.

    Args:
        D: hidden dimension (unused here, kept for API compatibility)
        split_threshold: coefficient of variation threshold for splitting.
            Higher → coarser selection. Lower → finer selection. Default 0.5.
        eps: numerical stability epsilon.
    """

    def __init__(
        self,
        D: int,
        split_threshold: float = 0.5,
        softmax_temperature: float = 0.3,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.builder = QuadTreeBuilder()
        self.navigator = QuadTreeNavigator(
            split_threshold=split_threshold,
            softmax_temperature=softmax_temperature,
            eps=eps,
        )