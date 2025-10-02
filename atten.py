"""
Attention Loss Functions for Video Action Recognition
Implements spatial alignment and temporal consistency losses
"""
import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path

_JSON_CACHE = {}

def _load_json_cached(path_str: str):
    """Load and cache JSON annotation files."""
    if not path_str or not os.path.isfile(path_str):
        return None
    if not path_str.lower().endswith(".json"):
        return None
    if os.path.getsize(path_str) == 0:
        return None
    if path_str in _JSON_CACHE:
        return _JSON_CACHE[path_str]
    try:
        with open(path_str, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        _JSON_CACHE[path_str] = data
        return data
    except Exception:
        return None


def _get_coords_for_frame(video_dict, frame_id: str):
    """
    Extract spatial coordinates from frame annotations.
    Supports multiple annotation formats.
    """
    frame_payload = video_dict.get(frame_id, {})
    if not isinstance(frame_payload, dict):
        return tuple()

    # Try different annotation field names
    loc = frame_payload.get("localization", None)

    if loc is None:
        text_field = frame_payload.get("text", None)
        if isinstance(text_field, str):
            try:
                parsed = json.loads(text_field)
                if isinstance(parsed, dict):
                    loc = parsed
            except Exception:
                return tuple()

    if loc is None:
        annotation_field = frame_payload.get("annotation", None)
        if isinstance(annotation_field, (dict, list)):
            loc = annotation_field

    if loc is None:
        bbox_field = frame_payload.get("bbox", None)
        if isinstance(bbox_field, (dict, list)):
            loc = bbox_field

    # Parse stringified JSON
    if isinstance(loc, str):
        loc_str = loc.strip()
        if loc_str:
            try:
                loc = json.loads(loc_str)
            except Exception:
                return tuple()
        else:
            return tuple()

    coords = []

    if isinstance(loc, dict):
        for coords_list in loc.values():
            if isinstance(coords_list, list):
                for c in coords_list:
                    if isinstance(c, (list, tuple)) and len(c) >= 2:
                        r, k = c[0], c[1]
                        if isinstance(r, (int, float)) and isinstance(k, (int, float)):
                            coords.append((int(r), int(k)))
    elif isinstance(loc, list):
        for c in loc:
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                r, k = c[0], c[1]
                if isinstance(r, (int, float)) and isinstance(k, (int, float)):
                    coords.append((int(r), int(k)))
            elif isinstance(c, dict):
                x = c.get("x", c.get("col", None))
                y = c.get("y", c.get("row", None))
                if x is not None and y is not None:
                    coords.append((int(y), int(x)))

    return tuple(coords)


def _coords_to_index(coords, grid_w: int):
    """Convert (row, col) coordinates to flattened indices."""
    return [r * grid_w + c for (r, c) in coords]


def build_gt_mask_from_batch_meta(metas, T: int, grid_h: int, grid_w: int):
    """
    Build ground truth spatial mask from metadata.

    Args:
        metas: List of metadata dictionaries
        T: Number of temporal frames
        grid_h, grid_w: Spatial grid dimensions

    Returns:
        M: [B,T,N] Ground truth mask (1 where annotation exists)
        Fmask: [B,T] Frame validity mask
        has_gt: [B] Sample validity mask
    """
    B = len(metas)
    N = grid_h * grid_w
    M = torch.zeros((B, T, N), dtype=torch.float32)
    Fmask = torch.zeros((B, T), dtype=torch.bool)
    has_gt = torch.zeros((B,), dtype=torch.bool)

    for b, meta in enumerate(metas):
        anno_path = meta.get("annotation_path")
        data = _load_json_cached(anno_path)
        if data is None:
            continue

        vid = meta.get("video_id")
        if not vid:
            continue

        # Try multiple video ID formats
        if vid in data:
            video_dict = data[vid]
        elif f"{vid}.avi" in data:
            video_dict = data[f"{vid}.avi"]
        elif "." in vid and vid.rsplit(".", 1)[0] in data:
            video_dict = data[vid.rsplit(".", 1)[0]]
        else:
            vid_no_ext = vid.replace(".avi", "")
            if vid_no_ext in data:
                video_dict = data[vid_no_ext]
            else:
                continue

        start = int(meta.get("start", 0))

        for t in range(T):
            # Try different frame ID formats
            frame_ids = [
                f"{start + t:04d}",
                f"{start + t:03d}",
                f"{start + t}",
                f"frame_{start + t:04d}",
                f"frame_{start + t}"
            ]

            coords_found = False
            for fid in frame_ids:
                if fid in video_dict:
                    coords = _get_coords_for_frame(video_dict, fid)
                    if coords:
                        Fmask[b, t] = True
                        for k in _coords_to_index(coords, grid_w):
                            if 0 <= k < N:
                                M[b, t, k] = 1.0
                        coords_found = True
                        break

            if coords_found:
                continue

        has_gt[b] = Fmask[b].any()

    return M, Fmask, has_gt


def spatial_attn_to_probs(attn_spatial: torch.Tensor, B: int, T: int, N: int, reduce: str = "mean"):
    """Convert spatial attention to probability distribution over patches."""
    BT, Hh, S, _ = attn_spatial.shape
    assert S == 1 + N
    cls2patch = attn_spatial[:, :, 0, 1:]  # [BT,Hh,N]
    if reduce == "mean":
        p = cls2patch.mean(dim=1)
    elif reduce == "max":
        p, _ = cls2patch.max(dim=1)
    else:
        raise ValueError("reduce must be 'mean' or 'max'")
    p = p.view(B, T, N)
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-12)
    return p


def spatial_kl_loss(p: torch.Tensor, M: torch.Tensor,
                   sample_mask: torch.Tensor, frame_mask: torch.Tensor,
                   eps: float = 1e-8):
    """
    Spatial alignment loss using KL divergence.

    Args:
        p: [B,T,N] Predicted attention probabilities
        M: [B,T,N] Ground truth spatial mask
        sample_mask: [B] Sample validity mask
        frame_mask: [B,T] Frame validity mask
        eps: Small constant for numerical stability

    Returns:
        KL divergence loss (scalar)
    """
    sm = sample_mask.bool()
    p = p[sm]
    M = M[sm]
    fm = frame_mask[sm]

    if p.numel() == 0:
        return p.new_tensor(0.0)

    pos_count = M.sum(dim=-1)
    valid = fm & (pos_count > 0)

    if not valid.any():
        return p.new_tensor(0.0)

    N = M.shape[-1]
    K = p.shape[-1]

    # Handle size mismatch
    if K != N:
        if K < N:
            pad_size = N - K
            p_padded = torch.cat([p, torch.zeros(p.shape[0], p.shape[1], pad_size, device=p.device)], dim=-1)
            p = p_padded
        else:
            p = p[:, :, :N]

    losses = []

    for b in range(p.shape[0]):
        for t in range(p.shape[1]):
            if not valid[b, t]:
                continue

            p_curr = p[b, t]
            m_curr = M[b, t]

            if p_curr.sum() < eps or m_curr.sum() < eps:
                continue

            # Normalize to probability distributions
            p_norm = p_curr / (p_curr.sum() + eps)
            m_norm = m_curr / (m_curr.sum() + eps)

            # KL divergence: KL(GT || predicted)
            kl_loss = F.kl_div(torch.log(p_norm + eps), m_norm, reduction='sum')

            if not (torch.isnan(kl_loss) or torch.isinf(kl_loss)):
                losses.append(kl_loss)

    return torch.stack(losses).mean() if losses else p.new_tensor(0.0)


def compute_kl_loss_batch(pred_probs: torch.Tensor, gt_probs: torch.Tensor, eps: float = 1e-8):
    """Compute KL divergence loss for a batch of probability distributions."""
    if pred_probs.numel() == 0 or gt_probs.numel() == 0:
        return 0.0

    losses = []
    B = pred_probs.shape[0]

    for b in range(B):
        p_pred = pred_probs[b]
        p_gt = gt_probs[b]

        p_pred_sum = p_pred.sum()
        p_gt_sum = p_gt.sum()

        if p_pred_sum < eps or p_gt_sum < eps:
            continue

        p_pred_norm = p_pred / (p_pred_sum + eps)
        p_gt_norm = p_gt / (p_gt_sum + eps)

        kl_loss = F.kl_div(torch.log(p_pred_norm + eps), p_gt_norm, reduction='sum')

        if not (torch.isnan(kl_loss) or torch.isinf(kl_loss)):
            losses.append(kl_loss)

    return torch.stack(losses).mean() if losses else pred_probs.new_tensor(0.0)
