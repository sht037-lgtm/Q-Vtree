from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """A node in the recursive partition tree."""
    node_id: int
    level: int
    region: Region
    parent: Optional[int]
    children: Optional[Tuple[int, ...]]


# =============================
# 2) Full tree builder
# =============================
class QuadTreeBuilder:
    """
    Build a FULL recursive partition tree down to 1x1 patch leaves.
    """

    @staticmethod
    def validate_hw(N: int, H: int, W: int) -> Tuple[int, int]:
        H = int(H)
        W = int(W)
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
            return (
                Region(reg.r0, rm, reg.c0, reg.c1),
                Region(rm, reg.r1, reg.c0, reg.c1),
            )

        if h == 1 and w >= 2:
            cm = (reg.c0 + reg.c1) // 2
            if cm == reg.c0 or cm == reg.c1:
                return None
            return (
                Region(reg.r0, reg.r1, reg.c0, cm),
                Region(reg.r0, reg.r1, cm, reg.c1),
            )

        return None

    @torch.no_grad()
    def build(self, x: torch.Tensor, H: int, W: int) -> Dict[str, Any]:
        B, N, D = x.shape
        H, W = self.validate_hw(N, H, W)

        nodes: List[Node] = []
        root_reg = Region(0, H, 0, W)
        nodes.append(Node(node_id=0, level=0, region=root_reg, parent=None, children=None))

        next_id = 1
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
                nodes.append(
                    Node(
                        node_id=cid,
                        level=nodes[pid].level + 1,
                        region=reg,
                        parent=pid,
                        children=None,
                    )
                )
                child_ids.append(cid)

            nodes[pid].children = tuple(child_ids)
            queue.extend(child_ids)

        return {"H": H, "W": W, "nodes": nodes}


# =============================
# 3) Attention-matrix scorer utils
# =============================
class AttentionMatrixScorer(nn.Module):
    """
    Utilities for query-conditioned visual scoring based on a text->vision
    attention matrix P of shape [B, Lt, Lv].
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    @torch.no_grad()
    def select_raters(self, score_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            score_matrix: [B, Lt, Lv], text->vision attention matrix.

        Returns:
            rater_mask: [B, Lt] bool
            text_scores: [B, Lt]
        """
        if score_matrix.dim() != 3:
            raise ValueError(
                f"score_matrix must be [B,Lt,Lv], got shape {tuple(score_matrix.shape)}"
            )

        P = torch.nan_to_num(score_matrix.float())
        text_scores = P.mean(dim=-1)  # [B, Lt]
        threshold = text_scores.mean(dim=-1, keepdim=True)
        rater_mask = text_scores >= threshold

        # fallback: if every token is filtered out due to degenerate values, keep all
        empty_rows = ~rater_mask.any(dim=-1)
        if empty_rows.any():
            rater_mask[empty_rows] = True

        return rater_mask, text_scores

    @torch.no_grad()
    def aggregate_visual_scores(
        self,
        score_matrix: torch.Tensor,
        rater_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            score_matrix: [B, Lt, Lv]
            rater_mask:  [B, Lt] or None

        Returns:
            patch_scores: [B, Lv]
        """
        if score_matrix.dim() != 3:
            raise ValueError(
                f"score_matrix must be [B,Lt,Lv], got shape {tuple(score_matrix.shape)}"
            )

        P = torch.nan_to_num(score_matrix.float())
        B, Lt, Lv = P.shape

        if rater_mask is None:
            rater_mask, _ = self.select_raters(P)

        patch_scores = []
        for b in range(B):
            selected = P[b][rater_mask[b]]
            if selected.numel() == 0:
                selected = P[b]
            score = selected.mean(dim=0)
            patch_scores.append(score)

        patch_scores = torch.stack(patch_scores, dim=0)  # [B, Lv]
        patch_scores = torch.nan_to_num(patch_scores)

        min_vals = patch_scores.min(dim=1, keepdim=True).values
        max_vals = patch_scores.max(dim=1, keepdim=True).values
        patch_scores = (patch_scores - min_vals) / (max_vals - min_vals + self.eps)
        return patch_scores.to(score_matrix.dtype)

    @torch.no_grad()
    def forward(self, score_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        rater_mask, text_scores = self.select_raters(score_matrix)
        patch_scores = self.aggregate_visual_scores(score_matrix, rater_mask=rater_mask)
        return {
            "score_matrix": score_matrix,
            "rater_mask": rater_mask,
            "text_scores": text_scores,
            "patch_scores": patch_scores,
        }


# =============================
# 4) Navigator
# =============================
class QuadTreeNavigator:
    """Tree logic based on a precomputed patch score map."""

    def __init__(self, split_threshold: float = 0.1, softmax_temperature: float = 1.0, eps: float = 1e-6):
        super().__init__()
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
        x = vals / self.softmax_temperature
        weights = torch.softmax(x, dim=0)
        return torch.sum(weights * vals)

    @torch.no_grad()
    def select_nodes(self, nodes: List[Node], patch_scores: torch.Tensor, W: int):
        B, N = patch_scores.shape
        weights = torch.softmax(patch_scores / self.softmax_temperature, dim=1)
        global_soft = (weights * patch_scores).sum(dim=1)

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

                s_soft = self._softmax_pool(vals)
                s_avg = vals.mean()

                if s_soft < global_soft[b]:
                    continue

                children = nodes[pid].children
                if not children:
                    selected[b].append(pid)
                    continue

                split_score = (s_soft - s_avg) / (s_avg + self.eps)
                if split_score > self.split_threshold:
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
    End-to-end module.

    Preferred usage:
      - pass precomputed patch_scores [B, N]
    Optional fallback:
      - pass score_matrix [B, Lt, N] and let the scorer aggregate it
    """

    def __init__(
        self,
        D: int,
        Dq: Optional[int] = None,
        use_proj_if_needed: bool = True,
        split_threshold: float = 0.1,
        softmax_temperature: float = 0.25,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.builder = QuadTreeBuilder()
        self.navigator = QuadTreeNavigator(
            split_threshold=split_threshold,
            softmax_temperature=softmax_temperature,
            eps=eps,
        )
        self.scorer = AttentionMatrixScorer(eps=eps)
        self.eps = float(eps)

        self._debug_score_matrix = None
        self._debug_rater_mask = None
        self._debug_text_scores = None
        self._debug_patch_scores = None

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        patch_scores: Optional[torch.Tensor] = None,
        score_matrix: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if x.dim() != 3:
            raise ValueError(f"x must be [B,N,D], got shape {tuple(x.shape)}")

        B, N, D = x.shape
        H = int(H)
        W = int(W)
        if H <= 0 or W <= 0:
            raise ValueError(f"H and W must be positive, got H={H}, W={W}")
        if H * W != N:
            raise ValueError(f"Invalid grid size: H*W={H*W}, but N={N}")

        if patch_scores is None:
            if score_matrix is None:
                raise ValueError("Either patch_scores or score_matrix must be provided.")
            scorer_out = self.scorer(score_matrix)
            patch_scores = scorer_out["patch_scores"]
            self._debug_score_matrix = scorer_out["score_matrix"]
            self._debug_rater_mask = scorer_out["rater_mask"]
            self._debug_text_scores = scorer_out["text_scores"]
        else:
            patch_scores = torch.nan_to_num(patch_scores)
            if patch_scores.dim() != 2 or patch_scores.shape != (B, N):
                raise ValueError(
                    f"patch_scores must be [B,N]={B,N}, got {tuple(patch_scores.shape)}"
                )
            self._debug_score_matrix = score_matrix
            self._debug_rater_mask = None
            self._debug_text_scores = None

        self._debug_patch_scores = patch_scores
        score_map = patch_scores.view(B, H, W)

        built = self.builder.build(x, H, W)
        H_tree, W_tree, nodes = built["H"], built["W"], built["nodes"]
        if H_tree != H or W_tree != W:
            raise ValueError(
                f"Patch-grid mismatch between x and score_map: tree inferred {(H_tree, W_tree)} but score_map is {(H, W)}"
            )

        patch_scores_flat = score_map.view(B, H * W)
        selected_node_ids, visited_node_ids = self.navigator.select_nodes(
            nodes=nodes,
            patch_scores=patch_scores_flat,
            W=W,
        )

        selected_regions = [[nodes[nid].region for nid in row] for row in selected_node_ids]
        token_out = self.navigator.nodes_to_tokens(
            nodes=nodes,
            H=H,
            W=W,
            selected_node_ids=selected_node_ids,
            x=x,
        )

        return {
            "H": H,
            "W": W,
            "nodes": nodes,
            "num_nodes": len(nodes),
            "patch_scores": patch_scores_flat,
            "score_matrix": score_matrix,
            "selected_node_ids": selected_node_ids,
            "selected_regions": selected_regions,
            "visited_node_ids": visited_node_ids,
            **token_out,
        }
