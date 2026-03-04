# qvtree.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque



# =============================
# 1) Data structures
# =============================
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
    """A node in the quadtree."""
    node_id: int
    level: int
    region: Region
    parent: Optional[int]
    children: Optional[Tuple[int, int, int, int]]  # (tl, tr, bl, br)
    feat: torch.Tensor  # [B, D] pooled feature


# =============================
# 2) Scoring (pluggable)
# =============================

class BaseScorer(nn.Module):
    """Interface: score(q, f) -> [B]."""
    def forward(self, q: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CosineScorer(nn.Module):
    """
    Cosine similarity scorer.
    Assumes q and f are already aligned: q.shape[-1] == f.shape[-1].
    No projection. If mismatched, raise error.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, q: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        q: [B, D]
        f: [B, D]
        return: [B]
        """
        if q.shape[-1] != f.shape[-1]:
            raise ValueError(f"Expected aligned dims, but got q:{q.shape[-1]} and f:{f.shape[-1]}")

        q = q / (q.norm(dim=-1, keepdim=True) + self.eps)
        f = f / (f.norm(dim=-1, keepdim=True) + self.eps)
        return (q * f).sum(dim=-1)


class DotProductScorer(nn.Module):
    """
    Dot-product similarity scorer.
    Assumes q and f are already aligned: q.shape[-1] == f.shape[-1].
    """

    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """
        q: [B, D]
        f: [B, D]
        return: [B]
        """
        if q.shape[-1] != f.shape[-1]:
            raise ValueError(
                f"Expected aligned dims, but got q:{q.shape[-1]} and f:{f.shape[-1]}"
            )

        return (q * f).sum(dim=-1)


# =============================
# 3) Full quadtree builder
# =============================
class QuadTreeBuilder:
    """
    Build a FULL quadtree down to 1x1 patch leaves.

    Input:
      x: [B, N, D], where N = H*W, and (assumed) H=W (square patch grid).

    Output:
      {"H": H, "W": W, "nodes": List[Node]}
    """

    def __init__(self, require_square_grid: bool = True):
        self.require_square_grid = bool(require_square_grid)

    @staticmethod
    def infer_hw(N: int, require_square_grid: bool = True) -> Tuple[int, int]:
        """N -> H and W"""
        if require_square_grid:
            H = int(math.isqrt(N))
            if H * H != N:
                raise ValueError(f"N={N} is not a perfect square, cannot infer H=W.")
            return H, H
        raise NotImplementedError("Non-square grid not supported in this version.")

    @staticmethod
    def tokens_to_grid(x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """x: [B,N,D] -> [B,H,W,D]"""
        B, N, D = x.shape
        assert N == H * W, f"N must equal H*W, got N={N}, H*W={H*W}"
        return x.view(B, H, W, D)

    @staticmethod
    def build_prefix_sum(grid: torch.Tensor) -> torch.Tensor:
        """
        grid: [B,H,W,D] -> prefix: [B,H+1,W+1,D]
        allows O(1) region sum.
        """
        B, H, W, D = grid.shape
        prefix = torch.zeros((B, H + 1, W + 1, D), device=grid.device, dtype=grid.dtype)
        prefix[:, 1:, 1:, :] = grid
        prefix = prefix.cumsum(dim=1).cumsum(dim=2)
        return prefix

    @staticmethod
    def region_sum(prefix: torch.Tensor, reg: Region) -> torch.Tensor:
        """Sum over region -> [B,D]"""
        r0, r1, c0, c1 = reg.r0, reg.r1, reg.c0, reg.c1
        return (
            prefix[:, r1, c1, :]
            - prefix[:, r0, c1, :]
            - prefix[:, r1, c0, :]
            + prefix[:, r0, c0, :]
        )

    @classmethod
    def region_mean(cls, prefix: torch.Tensor, reg: Region) -> torch.Tensor:
        """Average Pooling: mean over region -> [B,D]"""
        return cls.region_sum(prefix, reg) / float(reg.area)

    @staticmethod
    def can_split(reg: Region) -> bool:
        """Leaf is 1x1 patch."""
        return reg.h >= 2 and reg.w >= 2

    @staticmethod
    def split4(reg: Region) -> Optional[Tuple[Region, Region, Region, Region]]:
        """
        Integer midpoint split into 4 children (uneven sizes allowed for odd dimensions).
        Returns (tl,tr,bl,br) or None if cannot split.
        """
        if reg.h < 2 or reg.w < 2:
            return None
        rm = (reg.r0 + reg.r1) // 2
        cm = (reg.c0 + reg.c1) // 2

        if rm == reg.r0 or rm == reg.r1 or cm == reg.c0 or cm == reg.c1:
            return None

        tl = Region(reg.r0, rm, reg.c0, cm)
        tr = Region(reg.r0, rm, cm, reg.c1)
        bl = Region(rm, reg.r1, reg.c0, cm)
        br = Region(rm, reg.r1, cm, reg.c1)
        return tl, tr, bl, br

    @torch.no_grad()
    def build(self, x: torch.Tensor) -> Dict[str, Any]:
        B, N, D = x.shape
        H, W = self.infer_hw(N, self.require_square_grid)

        grid = self.tokens_to_grid(x, H, W)            # [B,H,W,D]
        prefix = self.build_prefix_sum(grid)           # [B,H+1,W+1,D]

        nodes: List[Node] = []
        next_id = 0

        # root
        root_reg = Region(0, H, 0, W)
        root_feat = self.region_mean(prefix, root_reg)  # [B,D]
        nodes.append(Node(
            node_id=next_id,
            level=0,
            region=root_reg,
            parent=None,
            children=None,
            feat=root_feat,
        ))
        next_id += 1

        # BFS to build FULL tree
        queue = deque([0])  # Optimization: O(n) -> O(1)
        while queue:
            pid = queue.popleft()
            preg = nodes[pid].region

            if not self.can_split(preg):
                continue

            sub = self.split4(preg)
            if sub is None:
                continue

            child_ids = []
            for reg in sub:
                feat = self.region_mean(prefix, reg)
                cid = next_id
                next_id += 1
                nodes.append(Node(
                    node_id=cid,
                    level=nodes[pid].level + 1,
                    region=reg,
                    parent=pid,
                    children=None,
                    feat=feat,
                ))
                child_ids.append(cid)

            nodes[pid].children = (child_ids[0], child_ids[1], child_ids[2], child_ids[3])
            queue.extend(child_ids)

        return {"H": H, "W": W, "nodes": nodes}


# =============================
# 4) Navigator
# =============================
class QuadTreeNavigator:
    """
    Implement Algorithm 1: multi-branch synchronous zoom-in:

      Q <- {root}
      S <- empty
      while Q not empty:
        pop p
        K <- { c in children(p) | s(c,q) > s(p,q) }
        if K empty: S <- S U {p}
        else: Q <- Q U K
      return S

    Output S is a variable-size set of nodes per sample.
    """

    def __init__(self, scorer: BaseScorer):
        super().__init__()
        self.scorer = scorer

    @staticmethod
    def region_to_token_indices(reg: Region, W: int, device: torch.device) -> torch.Tensor:
        """Flatten: idx = r*W + c. Return 1D indices of length reg.area."""
        rows = torch.arange(reg.r0, reg.r1, device=device)
        cols = torch.arange(reg.c0, reg.c1, device=device)
        rr, cc = torch.meshgrid(rows, cols, indexing="ij")
        return (rr * W + cc).reshape(-1)

    # there is a problem about ".item()"
    @torch.no_grad()
    def select_nodes(self, nodes: List[Node], q: torch.Tensor) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Per-sample navigation to keep logic exact.
        Return: selected_node_ids[b] = list of node ids in S for sample b.
        """
        B = q.shape[0]
        selected: List[List[int]] = [[] for _ in range(B)]
        visited = [[] for _ in range(B)]

        # stack feats for easy indexing: [M,B,D]
        feats = torch.stack([n.feat for n in nodes], dim=0)

        for b in range(B):
            qb = q[b:b+1]  # [1,Dq] , query
            Q = deque([0])  # root

            while Q:
                pid = Q.popleft()  # Optimization: O(n) -> O(1)
                visited[b].append(pid)

                pfeat = feats[pid, b:b+1, :]  # [1,D]
                sp = float(self.scorer(qb, pfeat).item())

                ch = nodes[pid].children
                if ch is None:
                    # leaf
                    selected[b].append(pid)
                    continue

                better_children = []
                for cid in ch:
                    cfeat = feats[cid, b:b+1, :]
                    sc = float(self.scorer(qb, cfeat).item())
                    if sc > sp:
                        better_children.append(cid)

                if len(better_children) == 0:
                    selected[b].append(pid)
                else:
                    Q.extend(better_children)

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
        """
        Convert selected nodes S into selected patch tokens.
        Returns:
          - selected_token_indices: list of 1D tensors (variable length)
          - if x provided: selected_feats_padded [B,Mmax,D] and selected_mask [B,Mmax]
        """
        device = x.device if x is not None else torch.device("cpu")
        B = len(selected_node_ids)

        token_indices: List[torch.Tensor] = []
        for b in range(B):
            idxs = []
            for nid in selected_node_ids[b]:
                reg = nodes[nid].region
                idxs.append(self.region_to_token_indices(reg, W=W, device=device))
            if len(idxs) == 0:
                all_idx = torch.empty(0, device=device, dtype=torch.long)
            else:
                all_idx = torch.cat(idxs, dim=0).unique(sorted=True)
            token_indices.append(all_idx)

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


# =============================
# 5) Plug-and-play wrapper
# =============================
class QVTree(nn.Module):
    """
    End-to-end module:
      input: x [B,N,D], q [B,Dq]
      1) build full quadtree to 1x1
      2) run Algorithm 1 navigation to select node set S
      3) return node-set + token-set (gathered from x)
    """

    def __init__(self, D: int, Dq: Optional[int] = None, use_proj_if_needed: bool = True):
        super().__init__()
        self.builder = QuadTreeBuilder(require_square_grid=True)
        self.scorer = DotProductScorer()
        self.navigator = QuadTreeNavigator(self.scorer)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, q: torch.Tensor) -> Dict[str, Any]:
        built = self.builder.build(x)
        H, W, nodes = built["H"], built["W"], built["nodes"]

        selected_node_ids, visited_node_ids = self.navigator.select_nodes(nodes, q)
        selected_regions = [[nodes[nid].region for nid in row] for row in selected_node_ids]

        token_out = self.navigator.nodes_to_tokens(nodes, H, W, selected_node_ids, x=x)

        return {
            "H": H,
            "W": W,
            "nodes":nodes,
            "num_nodes": len(nodes),
            "selected_node_ids": selected_node_ids,
            "selected_regions": selected_regions,
            "visited_node_ids": visited_node_ids,
            **token_out,
        }

def region_to_patch_ids(region: Region, grid_w: int):
    ids = []
    for r in range(region.r0, region.r1):
        for c in range(region.c0, region.c1):
            ids.append(r * grid_w + c)
    return ids