"""
ImitationNet reproduction — evaluate retargeting (global-rotation / faithful version).

Human (4-limb body-canonical 6D input) -> Q_h -> latent -> D_r -> robot joint angles -> FK -> robot limb
GLOBAL rotations. Faithful S_RD compares human vs retargeted-robot limb rotations.
Lower S_RD and per-limb geodesic angle = better; compared against a RANDOM robot baseline.
"""
import torch
from robot_kinematics import RobotFK
from human_repr import load_humanml3d_poses, human_limb_frames_canon
from rotations import s_rd_rot, rotmat_to_6d
from imitationnet_model import ImitationNet
from paths import ckpt

dev = "cpu"
fk = RobotFK(device=dev)
_c = torch.load(ckpt("best"), map_location=dev)
model = ImitationNet(latent=_c.get("latent", 8), hidden=_c.get("hidden", 128)).to(dev)
model.load_state_dict(_c["model"])
model.eval()

J = load_humanml3d_poses("test", max_motions=300, stride=10)
Rh = human_limb_frames_canon(J).to(dev)        # (N,4,3,3) human body-canonical limb frames
h_in = rotmat_to_6d(Rh).reshape(Rh.shape[0], -1)  # (N,24) input
N = h_in.shape[0]
print(f"eval on {N} human test poses (body-canonical rotation S_RD)")

with torch.no_grad():
    xr_norm = model.retarget(h_in)             # (N,14) normalized joint angles
    xr_real = fk.denormalize(xr_norm)
    Rr = fk.limb_frames_canon(xr_real)         # (N,4,3,3) retargeted robot canonical frames
    s_ret = s_rd_rot(Rh, Rr)                   # (N,) human vs retargeted robot
    Rrand = fk.limb_frames_canon(fk.sample_arm_poses(N))
    s_rand = s_rd_rot(Rh, Rrand)

print(f"\nS_RD (lower=better, 0..4):")
print(f"  retargeted vs human : mean {s_ret.mean():.3f}  median {s_ret.median():.3f}")
print(f"  RANDOM     vs human : mean {s_rand.mean():.3f}  median {s_rand.median():.3f}")
print(f"  -> improvement factor: {s_rand.mean()/s_ret.mean():.2f}x")

# per-limb geodesic rotation error (deg)
Rrel = Rh.transpose(-1, -2) @ Rr
cos = ((Rrel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
ang = torch.rad2deg(torch.acos(cos))           # (N,4)
print(f"\nper-limb rotation error (deg): mean {ang.mean():.1f}  "
      f"[L_up {ang[:,0].mean():.0f}, L_lo {ang[:,1].mean():.0f}, R_up {ang[:,2].mean():.0f}, R_lo {ang[:,3].mean():.0f}]")
