"""
Rotation utilities for the FAITHFUL (global-rotation) ImitationNet similarity.

The paper's S_RD compares the GLOBAL ROTATION of limbs (Eq.1), capturing full 3D
orientation incl. twist — not just direction. We extract limb rotation MATRICES for
human (FK-compose HumanML3D rot6d, root-relative) and robot (FK link rotations), and
compute S_RD directly from matrices:

    1 - <q_h,q_r>^2 = (1 - cos theta)/2,   cos theta = (trace(R_h^T R_r) - 1)/2

(theta = geodesic angle between the two rotations; <q_h,q_r> = cos(theta/2)). This
avoids quaternion double-cover sign issues.
"""
import torch


def sixd_to_rotmat(d6):
    """(...,6) continuous 6D rotation (Zhou et al.) -> (...,3,3) rotation matrix."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)          # columns b1,b2,b3


def rotmat_to_6d(R):
    """(...,3,3) -> (...,6): first two columns (network-friendly rotation repr)."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def _normalize(v, eps=1e-8):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def limb_frame(prox, dist, ref):
    """Build a limb's GLOBAL rotation frame from 3 joint positions (all (...,3)):
      x = bone direction (prox->dist);  y = arm-plane (ref bone, orthogonalized);  z = x*y.
    Captures direction + arm-plane orientation (the observable 'twist'). Returns (...,3,3)."""
    x = _normalize(dist - prox)
    r = ref - dist
    y = _normalize(r - (r * x).sum(-1, keepdim=True) * x)
    z = torch.cross(x, y, dim=-1)
    return torch.stack([x, y, z], dim=-1)             # columns x,y,z


def body_frame(l_shoulder, r_shoulder):
    """Torso frame from the two shoulder positions (z = world up). Returns (...,3,3).
    Used to canonicalize limb rotations into a body-relative (coord-system-invariant) frame."""
    up = torch.zeros_like(l_shoulder); up[..., 2] = 1.0
    left = _normalize(l_shoulder - r_shoulder)
    fwd = _normalize(torch.cross(left, up, dim=-1))
    left = torch.cross(up, fwd, dim=-1)               # re-orthonormalize
    return torch.stack([fwd, left, up], dim=-1)       # columns fwd,left,up


def canonicalize(R_world, B):
    """Express world limb frames R_world (...,4,3,3) in body frame B (...,3,3):
    R_canon = B^T @ R_world. World-frame-invariant -> human & robot become comparable."""
    return B.transpose(-1, -2).unsqueeze(-3) @ R_world


def s_rd_rot(Rh, Rr):
    """
    Rh,Rr: (B,4,3,3) limb rotation matrices for the 4 limbs.
    Returns (B,) S_RD = sum_limb (1 - cos theta)/2,  cos theta=(trace(Rh^T Rr)-1)/2.
    """
    Rrel = Rh.transpose(-1, -2) @ Rr                  # (B,4,3,3)
    tr = Rrel.diagonal(dim1=-2, dim2=-1).sum(-1)      # (B,4) trace
    cos = ((tr - 1.0) * 0.5).clamp(-1, 1)
    return ((1.0 - cos) * 0.5).sum(-1)                # sum over 4 limbs


def s_rd_rot_pairwise(Rh, Rr):
    """Rh:(H,4,3,3), Rr:(R,4,3,3) -> (H,R) pairwise S_RD."""
    # Rrel[h,r,l] = Rh[h,l]^T @ Rr[r,l]; trace via einsum on Rh^T and Rr
    Rht = Rh.transpose(-1, -2)                        # (H,4,3,3)
    tr = torch.einsum("hlij,rlji->hrl", Rht, Rr)      # (H,R,4) trace of product
    cos = ((tr - 1.0) * 0.5).clamp(-1, 1)
    return ((1.0 - cos) * 0.5).sum(-1)                # (H,R)
