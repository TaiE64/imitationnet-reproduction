"""
ImitationNet reproduction — robot kinematics (TIAGo++).

Provides: load URDF, sample robot poses (joint angles within limits), forward
kinematics, and extract the 4 arm-limb DIRECTION vectors used by the similarity
metric (paper: left/right shoulder->elbow and elbow->wrist).

The robot "pose" x_r = the 14 arm joint angles (left 7 + right 7). Other joints
(torso/head/wheels/grippers) are held at 0 — they don't change arm-limb directions.

URDF: datasets/robot_retargeting/tiago_dual.urdf (paper trains on TIAGo++).
"""
import os
import torch
import pytorch_kinematics as pk

URDF = "/home/taie/Downloads/Reproduce/datasets/robot_retargeting/tiago_dual.urdf"

# the 14 arm joints (left 1..7, right 1..7), in the chain's actuated-joint order
ARM_JOINTS = [f"arm_left_{i}_joint" for i in range(1, 8)] + \
             [f"arm_right_{i}_joint" for i in range(1, 8)]

# limb endpoints (link names): shoulder -> elbow -> wrist, per arm
LIMB_LINKS = {
    "L_upper": ("arm_left_2_link", "arm_left_4_link"),   # left shoulder->elbow
    "L_lower": ("arm_left_4_link", "arm_left_7_link"),   # left elbow->wrist
    "R_upper": ("arm_right_2_link", "arm_right_4_link"),
    "R_lower": ("arm_right_4_link", "arm_right_7_link"),
}
LIMB_ORDER = ["L_upper", "L_lower", "R_upper", "R_lower"]
N_ARM = len(ARM_JOINTS)   # 14


class RobotFK:
    def __init__(self, urdf=URDF, device="cpu"):
        self.device = device
        self.chain = pk.build_chain_from_urdf(open(urdf, "rb").read()).to(device=device)
        self.jnames = self.chain.get_joint_parameter_names()
        lo, hi = self.chain.get_joint_limits()
        self.lo = torch.as_tensor(lo, dtype=torch.float32, device=device)
        self.hi = torch.as_tensor(hi, dtype=torch.float32, device=device)
        # indices of the 14 arm joints within the full actuated vector
        self.arm_idx = torch.tensor([self.jnames.index(j) for j in ARM_JOINTS], device=device)

    def arm_limits(self):
        """(lo,hi) for the 14 arm joints."""
        return self.lo[self.arm_idx], self.hi[self.arm_idx]

    def normalize(self, pose):
        """(B,14) joint angles -> [-1,1] using arm joint limits."""
        lo, hi = self.arm_limits()
        return 2.0 * (pose - lo) / (hi - lo + 1e-8) - 1.0

    def denormalize(self, norm):
        """[-1,1] -> joint angles (rad)."""
        lo, hi = self.arm_limits()
        return (norm + 1.0) / 2.0 * (hi - lo) + lo

    def sample_arm_poses(self, n):
        """Sample n robot poses = (n,14) arm joint angles uniformly within limits."""
        lo, hi = self.arm_limits()
        return lo + (hi - lo) * torch.rand(n, N_ARM, device=self.device)

    def _full_config(self, arm_angles):
        """(B,14) arm angles -> (B, n_actuated) full config (others=0)."""
        B = arm_angles.shape[0]
        full = torch.zeros(B, len(self.jnames), device=self.device)
        full[:, self.arm_idx] = arm_angles
        return full

    def limb_dirs(self, arm_angles):
        """
        (B,14) arm angles -> (B,4,3) unit direction vectors of the 4 limbs,
        in order LIMB_ORDER. Direction = pos(child_link) - pos(parent_link).
        """
        full = self._full_config(arm_angles)
        fk = self.chain.forward_kinematics(full)       # dict link -> Transform3d
        pos = {k: v.get_matrix()[:, :3, 3] for k, v in fk.items()}  # (B,3) each
        dirs = []
        for limb in LIMB_ORDER:
            a, b = LIMB_LINKS[limb]
            d = pos[b] - pos[a]                         # (B,3)
            d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
            dirs.append(d)
        return torch.stack(dirs, dim=1)                # (B,4,3)

    def limb_rotmats(self, arm_angles):
        """
        (B,14) arm angles -> (B,4,3,3) GLOBAL rotation matrices of the 4 limbs (paper Eq.1).
        Limb rotation = global rotation of its PROXIMAL link (matches human shoulder/elbow).
        """
        full = self._full_config(arm_angles)
        fk = self.chain.forward_kinematics(full)
        rot = {k: v.get_matrix()[:, :3, :3] for k, v in fk.items()}  # (B,3,3) each
        return torch.stack([rot[LIMB_LINKS[l][0]] for l in LIMB_ORDER], dim=1)  # (B,4,3,3)

    def limb_frames_canon(self, arm_angles):
        """(B,14) -> (B,4,3,3) body-canonicalized limb rotation frames, built from link
        POSITIONS the SAME way as the human -> coordinate-system invariant & comparable."""
        from rotations import limb_frame, body_frame, canonicalize
        full = self._full_config(arm_angles)
        fk = self.chain.forward_kinematics(full)
        pos = {k: v.get_matrix()[:, :3, 3] for k, v in fk.items()}   # (B,3) each
        # limb -> (proximal, distal, reference) links (shoulder/elbow/wrist of each arm)
        frame_links = {
            "L_upper": ("arm_left_2_link", "arm_left_4_link", "arm_left_7_link"),
            "L_lower": ("arm_left_4_link", "arm_left_7_link", "arm_left_2_link"),
            "R_upper": ("arm_right_2_link", "arm_right_4_link", "arm_right_7_link"),
            "R_lower": ("arm_right_4_link", "arm_right_7_link", "arm_right_2_link"),
        }
        Rw = torch.stack([
            limb_frame(pos[p], pos[d], pos[r])
            for (p, d, r) in (frame_links[l] for l in LIMB_ORDER)
        ], dim=1)                                                    # (B,4,3,3)
        B = body_frame(pos["arm_left_2_link"], pos["arm_right_2_link"])
        return canonicalize(Rw, B)                                   # (B,4,3,3)


if __name__ == "__main__":
    fk = RobotFK()
    print("actuated joints:", len(fk.jnames), "| arm joints:", N_ARM)
    x = fk.sample_arm_poses(5)
    print("sampled arm poses:", tuple(x.shape))
    dirs = fk.limb_dirs(x)
    print("limb dirs:", tuple(dirs.shape), "(expect (5,4,3))")
    print("sample limb dirs (pose0):")
    for i, limb in enumerate(LIMB_ORDER):
        print(f"  {limb}: {dirs[0, i].numpy().round(3)}  |dir|={dirs[0,i].norm():.3f}")
