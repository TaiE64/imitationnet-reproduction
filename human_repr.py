"""
ImitationNet reproduction — human pose representation.

From the InterHuman 22-joint xyz skeleton we extract the SAME 4 arm-limb direction
vectors as the robot (left/right shoulder->elbow and elbow->wrist), so human and
robot limbs live in a comparable space for the similarity metric.

SMPL 22-joint arm indices: L shoulder16 -> L elbow18 -> L wrist20,
                           R shoulder17 -> R elbow19 -> R wrist21.
"""
import os
import numpy as np
import torch
from rotations import sixd_to_rotmat

# limb -> (parent_joint, child_joint), SAME order as robot_kinematics.LIMB_ORDER
HUMAN_LIMBS = {
    "L_upper": (16, 18), "L_lower": (18, 20),
    "R_upper": (17, 19), "R_lower": (19, 21),
}
LIMB_ORDER = ["L_upper", "L_lower", "R_upper", "R_lower"]
# proximal joint whose GLOBAL rotation = the limb's global rotation (paper Eq.1 uses
# the global rotation of body limbs): upper arm rotates with shoulder, forearm with elbow.
LIMB_PROXIMAL = {"L_upper": 16, "L_lower": 18, "R_upper": 17, "R_lower": 19}
# SMPL 22-joint parents (for FK composition of local rotations -> global)
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

DATA_ROOT = "/home/taie/Downloads/Reproduce/datasets/interhuman"
HML3D_ROOT = "/home/taie/Downloads/Reproduce/datasets/HumanML3D/HumanML3D"


def human_limb_dirs(joints):
    """joints (B,22,3) tensor -> (B,4,3) unit limb-direction vectors (LIMB_ORDER)."""
    dirs = []
    for limb in LIMB_ORDER:
        a, b = HUMAN_LIMBS[limb]
        d = joints[:, b] - joints[:, a]
        d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
        dirs.append(d)
    return torch.stack(dirs, dim=1)


# limb -> (proximal, distal, reference) joints for building the limb rotation frame
_LIMB_FRAME_J = {
    "L_upper": (16, 18, 20), "L_lower": (18, 20, 16),
    "R_upper": (17, 19, 21), "R_lower": (19, 21, 17),
}


def human_limb_frames_canon(joints):
    """(N,22,3) joint positions -> (N,4,3,3) body-canonicalized limb rotation frames.
    Coordinate-system invariant -> directly comparable to the robot's canonical frames."""
    from rotations import limb_frame, body_frame, canonicalize
    Rw = torch.stack([
        limb_frame(joints[:, p], joints[:, d], joints[:, r])
        for (p, d, r) in (_LIMB_FRAME_J[l] for l in LIMB_ORDER)
    ], dim=1)                                              # (N,4,3,3) world frames
    B = body_frame(joints[:, 16], joints[:, 17])           # torso frame from shoulders
    return canonicalize(Rw, B)                             # (N,4,3,3) body-relative


def load_human_poses(split="train", max_motions=None, stride=5):
    """Load human poses from InterHuman (both persons), every `stride` frames.
    Returns (N,22,3) tensor of poses (meters)."""
    p1 = os.path.join(DATA_ROOT, "motions_processed", "person1")
    p2 = os.path.join(DATA_ROOT, "motions_processed", "person2")
    ids = [l.strip() for l in open(os.path.join(DATA_ROOT, f"{split}.txt")) if l.strip()]
    poses = []
    for k, mid in enumerate(ids):
        if max_motions and k >= max_motions:
            break
        for d in (p1, p2):
            f = os.path.join(d, f"{mid}.npy")
            if os.path.exists(f):
                m = np.load(f).astype(np.float32)[:, :66].reshape(-1, 22, 3)
                poses.append(m[::stride])
    poses = np.concatenate(poses, axis=0)
    return torch.from_numpy(poses).float()


def human_global_rotmats(rot6d):
    """
    rot6d: (N,21,6) local 6D rotations of HumanML3D joints 1..21 (root=0 implicit identity).
    Returns (N,22,3,3) GLOBAL joint rotation matrices (root-relative), via FK composition
    along the SMPL parent tree.
    """
    N = rot6d.shape[0]
    local = sixd_to_rotmat(rot6d)                          # (N,21,3,3) for joints 1..21
    I = torch.eye(3, device=rot6d.device).expand(N, 3, 3)
    g = [I]                                                # joint 0 (root) = identity (root-relative)
    for j in range(1, 22):
        p = SMPL_PARENTS[j]
        g.append(g[p] @ local[:, j - 1])                  # global[j] = global[parent] @ local[j]
    return torch.stack(g, dim=1)                           # (N,22,3,3)


def human_limb_rotmats(rot6d):
    """(N,21,6) rot6d -> (N,4,3,3) GLOBAL rotation of the 4 limbs (paper Eq.1 basis)."""
    g = human_global_rotmats(rot6d)                        # (N,22,3,3)
    return torch.stack([g[:, LIMB_PROXIMAL[l]] for l in LIMB_ORDER], dim=1)  # (N,4,3,3)


# rot6d index for each limb's proximal joint (joint j -> rot6d index j-1)
_LIMB_ROT6D_IDX = [LIMB_PROXIMAL[l] - 1 for l in LIMB_ORDER]   # [15,17,16,18]


def human_limb_local6d(rot6d):
    """(N,21,6) -> (N,24): LOCAL 6D rotations of the 4 limb joints = the network INPUT x_h
    (paper §III-A: human joint as quaternion relative to parent)."""
    return rot6d[:, _LIMB_ROT6D_IDX].reshape(rot6d.shape[0], -1)   # (N,4,6)->(N,24)


def load_humanml3d_rot(split="train", max_motions=None, stride=5):
    """Load HumanML3D rot6d features (new_joint_vecs) -> (N,21,6) local rotations.
    Feature layout (263): root(4)+ric(63)+rot6d(126)+vel(66)+foot(4); rot6d at [67:193]."""
    ids = [l.strip() for l in open(os.path.join(HML3D_ROOT, f"{split}.txt")) if l.strip()]
    vdir = os.path.join(HML3D_ROOT, "new_joint_vecs")
    rots = []
    for k, mid in enumerate(ids):
        if max_motions and k >= max_motions:
            break
        f = os.path.join(vdir, f"{mid}.npy")
        if os.path.exists(f):
            v = np.load(f).astype(np.float32)              # (T,263)
            if v.ndim == 2 and v.shape[1] == 263:
                r = v[::stride, 67:67 + 126].reshape(-1, 21, 6)
                rots.append(r)
    rots = np.concatenate(rots, axis=0)
    return torch.from_numpy(rots).float()


def load_humanml3d_poses(split="train", max_motions=None, stride=5):
    """Load human poses from HumanML3D new_joints (paper's human dataset).
    Each motion is (T,22,3) in the same t2m/SMPL joint order. Returns (N,22,3)."""
    split_file = os.path.join(HML3D_ROOT, f"{split}.txt")
    ids = [l.strip() for l in open(split_file) if l.strip()]
    jdir = os.path.join(HML3D_ROOT, "new_joints")
    poses = []
    for k, mid in enumerate(ids):
        if max_motions and k >= max_motions:
            break
        f = os.path.join(jdir, f"{mid}.npy")
        if os.path.exists(f):
            m = np.load(f).astype(np.float32)        # (T,22,3)
            if m.ndim == 3 and m.shape[1] == 22:
                poses.append(m[::stride])
    poses = np.concatenate(poses, axis=0)
    return torch.from_numpy(poses).float()


if __name__ == "__main__":
    print("=== InterHuman ===")
    p = load_human_poses("train", max_motions=50)
    print("poses:", tuple(p.shape))
    print("=== HumanML3D ===")
    p2 = load_humanml3d_poses("train", max_motions=50)
    print("poses:", tuple(p2.shape))
    d = human_limb_dirs(p2[:3])
    for i, limb in enumerate(LIMB_ORDER):
        print(f"  {limb}: {d[0,i].numpy().round(3)}")
    p = p2  # below uses HumanML3D
    print("human poses:", tuple(p.shape))
    d = human_limb_dirs(p[:5])
    print("limb dirs:", tuple(d.shape), "(expect (5,4,3))")
    for i, limb in enumerate(LIMB_ORDER):
        print(f"  {limb}: {d[0,i].numpy().round(3)}  |dir|={d[0,i].norm():.3f}")
