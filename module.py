# module.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import math
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
    """A node in the quadtree."""
    node_id: int
    level: int
    region: Region
    parent: Optional[int]
    children: Optional[Tuple[int, int, int, int]]  # (tl, tr, bl, br)


# =============================
# 2) Full quadtree builder
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
        """Infer H and W from N."""
        if require_square_grid:
            H = int(math.isqrt(N))
            if H * H != N:
                raise ValueError(f"N={N} is not a perfect square, cannot infer H=W.")
            return H, H
        raise NotImplementedError("Non-square grid not supported in this version.")

    @staticmethod
    def can_split(reg: Region) -> bool:
        """Leaf is 1x1 patch."""
        return reg.h >= 2 and reg.w >= 2

    @staticmethod
    def split4(reg: Region) -> Optional[Tuple[Region, Region, Region, Region]]:
        """
        Integer midpoint split into 4 children (uneven sizes allowed for odd dimensions).
        Returns (tl, tr, bl, br) or None if cannot split.
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

        nodes: List[Node] = []
        next_id = 0

        # root
        root_reg = Region(0, H, 0, W)
        nodes.append(
            Node(
                node_id=next_id,
                level=0,
                region=root_reg,
                parent=None,
                children=None,
            )
        )
        next_id += 1

        # BFS to build FULL tree
        queue = deque([0])
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

            nodes[pid].children = (child_ids[0], child_ids[1], child_ids[2], child_ids[3])
            queue.extend(child_ids)

        return {"H": H, "W": W, "nodes": nodes}


# =============================
# 3) Attention Scorer
# =============================
class AttentionScorer(nn.Module):

    def __init__(self, eps=1e-6, temp=2):
        super().__init__()
        self.eps = eps
        self.temp = temp

    def forward(self, t, v):
        """
        t : [B, Lt, D]
        v : [B, Lv, D]
        return : [B, Lv]
        """

        # ---------- debug ----------
        # print("vision NaN:", torch.isnan(v).any())
        # print("text NaN:", torch.isnan(t).any())
        # print("vision Inf:", torch.isinf(v).any())

        dtype = v.dtype

        # ---------- clean + use fp32 ----------
        t = torch.nan_to_num(t).float()
        v = torch.nan_to_num(v).float()

        # ---------- normalize ----------
        t = F.normalize(t, dim=-1, eps=self.eps)
        v = F.normalize(v, dim=-1, eps=self.eps)

        # ---------- Vision → Text ----------
        S_vt = v @ t.transpose(-1, -2)  # [B, Lv, Lt]

        S_vt = S_vt / self.temp
        S_vt = S_vt - S_vt.max(dim=2, keepdim=True).values
        S_vt = torch.nan_to_num(S_vt)

        A_vt = torch.softmax(S_vt, dim=2)

        # ---------- text importance ----------
        text_score = A_vt.mean(dim=1)  # [B, Lt]

        scores = []

        for b in range(t.shape[0]):

            thresh = text_score[b].mean()
            rater_mask = text_score[b] >= thresh

            if rater_mask.sum() == 0:
                t_r = t[b]
            else:
                t_r = t[b][rater_mask]

            # ---------- Text → Vision ----------
            S_tv = v[b] @ t_r.T  # [Lv, Lr]

            S_tv = S_tv / self.temp
            S_tv = S_tv - S_tv.max(dim=0, keepdim=True).values
            S_tv = torch.nan_to_num(S_tv)

            A_tv = torch.softmax(S_tv, dim=0)

            vision_score = A_tv.mean(dim=1)  # [Lv]

            scores.append(vision_score)

        scores = torch.stack(scores)

        # ---------- min-max normalize ----------
        min_vals = scores.min(dim=1, keepdim=True).values
        max_vals = scores.max(dim=1, keepdim=True).values

        scores = (scores - min_vals) / (max_vals - min_vals + self.eps)

        # ---------- cast back ----------
        scores = scores.to(dtype)

        return scores


# =============================
# 4) Navigator
# =============================
class QuadTreeNavigator:
    """
    Tree logic based on a precomputed patch score map.

    For each node R:
      1) compute node softmax-pooled score over its patches
      2) discard node if softmax_score < global_image_average
      3) otherwise compute split score:
            (softmax_score - local_average_score) / (local_average_score + eps)
         if split_score > split_threshold and node is splittable:
            split and continue on children
         else:
            keep this node as a selected final node

    Navigation is performed independently for each sample in the batch.
    """

    def __init__(self, split_threshold: float = 0.1, softmax_temperature: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.split_threshold = float(split_threshold)
        self.softmax_temperature = float(softmax_temperature)
        self.eps = float(eps)

    @staticmethod
    def region_to_token_indices(reg: Region, W: int, device: torch.device) -> torch.Tensor:
        """Flatten: idx = r*W + c. Return 1D indices of length reg.area."""
        rows = torch.arange(reg.r0, reg.r1, device=device)
        cols = torch.arange(reg.c0, reg.c1, device=device)
        rr, cc = torch.meshgrid(rows, cols, indexing="ij")
        return (rr * W + cc).reshape(-1)

    def _softmax_pool(self, vals: torch.Tensor) -> torch.Tensor:
        """
        vals: [M]
        return: scalar tensor
        """
        # temperature fixed to 1 by default, but kept configurable
        x = vals / self.softmax_temperature
        if torch.isnan(x).any():
            print("softmax input has NaN")
        if torch.isinf(x).any():
            print("softmax input has Inf")

        weights = torch.softmax(x, dim=0)
        if torch.isnan(weights).any():
            print("softmax output has NaN")
        if torch.isinf(weights).any():
            print("softmax output has Inf")
        return torch.sum(weights * vals)

    @torch.no_grad()
    def select_nodes(
            self,
            nodes: List[Node],
            patch_scores: torch.Tensor,
            W: int,
    ):

        B, N = patch_scores.shape

        # global_avg = patch_scores.mean(dim=1)

        # global softmax pooling
        weights = torch.softmax(patch_scores / self.softmax_temperature, dim=1)
        global_soft = (weights * patch_scores).sum(dim=1)

        selected = [[] for _ in range(B)]
        visited = [[] for _ in range(B)]

        for b in range(B):

            Q = deque([0])

            while Q:

                pid = Q.popleft()
                visited[b].append(pid)

                # ---------- get region patches ----------
                reg = nodes[pid].region
                idx = self.region_to_token_indices(reg, W, patch_scores.device)
                vals = patch_scores[b, idx]

                # ---------- define scores ----------
                s_soft = self._softmax_pool(vals)
                s_avg = vals.mean()

                # ---------- discard ----------
                if s_soft < global_soft[b]:
                    continue

                # ---------- split ----------
                children = nodes[pid].children
                if not children:
                    selected[b].append(pid)
                    continue

                split_score = (s_soft - s_avg) / (s_avg + self.eps)
                print(f'split score: {split_score}')

                if split_score > self.split_threshold:
                    Q.extend(children)
                else:
                    selected[b].append(pid)

        print("patch_scores NaN:", torch.isnan(patch_scores).any())
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
        Convert selected nodes into selected patch tokens.

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
# 4) Plug-and-play wrapper
# =============================
class QVTree(nn.Module):
    """
    End-to-end module:
      input:
        x [B, N, D]   vision token features
        t [B, Lt, D]  text tokens

      1) compute patch score map using AttentionScorer
      2) build quadtree
      3) run navigation on the score map
      4) return selected node set + selected token set
    """

    def __init__(
        self,
        D: int,
        Dq: Optional[int] = None,
        use_proj_if_needed: bool = True,
        split_threshold: float = 0.1,
        softmax_temperature: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.builder = QuadTreeBuilder(require_square_grid=True)

        self.navigator = QuadTreeNavigator(
            split_threshold=split_threshold,
            softmax_temperature=softmax_temperature,
            eps=eps,
        )

        # get attention score map
        self.scorer = AttentionScorer()
        self.eps = float(eps)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            x : [B, N, D]   vision tokens
            t : [B, Lt, D]  text tokens
        """

        if x.dim() != 3:
            raise ValueError(f"x must be [B,N,D], got shape {tuple(x.shape)}")

        B, N, D = x.shape

        # --------------------------------------------------
        # compute patch scores using AttentionScorer
        # --------------------------------------------------

        patch_scores = self.scorer(t, x)  # [B, N]

        # infer grid
        grid = int(math.sqrt(N))
        if grid * grid != N:
            raise ValueError(f"Vision tokens must form square grid, got N={N}")

        H = grid
        W = grid

        score_map = patch_scores.view(B, H, W)

        # Debug store
        self._debug_patch_scores = patch_scores

        # --------------------------------------------------
        # build quadtree
        # --------------------------------------------------

        built = self.builder.build(x)
        H_tree, W_tree, nodes = built["H"], built["W"], built["nodes"]

        if H_tree != H or W_tree != W:
            raise ValueError(
                f"Patch-grid mismatch between x and score_map: "
                f"tree inferred {(H_tree, W_tree)} but score_map is {(H, W)}"
            )

        # flatten score map
        patch_scores = score_map.view(B, H * W)

        # --------------------------------------------------
        # tree navigation
        # --------------------------------------------------

        selected_node_ids, visited_node_ids = self.navigator.select_nodes(
            nodes=nodes,
            patch_scores=patch_scores,
            W=W,
        )

        # --------------------------------------------------
        # region info
        # --------------------------------------------------

        selected_regions = [
            [nodes[nid].region for nid in row]
            for row in selected_node_ids
        ]

        # --------------------------------------------------
        # convert nodes → tokens
        # --------------------------------------------------

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
            "patch_scores": patch_scores,
            "selected_node_ids": selected_node_ids,
            "selected_regions": selected_regions,
            "visited_node_ids": visited_node_ids,
            **token_out,
        }