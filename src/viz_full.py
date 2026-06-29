"""
Same retargeting, but draw the ROBOT's FULL 7-DOF arm chain (arm_1..7 links) vs the
human's 3-joint arm (shoulder-elbow-wrist). Shows the real kinematic difference.
"""
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from robot_kinematics import RobotFK
from human_repr import load_humanml3d_poses, human_limb_frames_canon
from rotations import rotmat_to_6d, body_frame
from imitationnet_model import ImitationNet
from paths import ckpt, media

fk = RobotFK()
_c = torch.load(ckpt("best"), map_location="cpu")
model = ImitationNet(latent=_c.get("latent", 8), hidden=_c.get("hidden", 128))
model.load_state_dict(_c["model"])
model.eval()

J = load_humanml3d_poses("test", max_motions=400, stride=15)
elev = (J[:, 20, 2] + J[:, 21, 2]) / 2 - (J[:, 16, 2] + J[:, 17, 2]) / 2
idx = torch.argsort(elev)[torch.linspace(0, len(elev) - 1, 6).long()]
Js = J[idx]
Rh = human_limb_frames_canon(Js)
h_in = rotmat_to_6d(Rh).reshape(Rh.shape[0], -1)
with torch.no_grad():
    xr = fk.denormalize(model.retarget(h_in))
rpos = {k: v.get_matrix()[:, :3, 3] for k, v in fk.chain.forward_kinematics(fk._full_config(xr)).items()}

LARM = [f"arm_left_{i}_link" for i in range(1, 8)]    # full 7-link chain
RARM = [f"arm_right_{i}_link" for i in range(1, 8)]


def canon(pts, lsh, rsh):
    B = body_frame(lsh, rsh); c = (lsh + rsh) / 2; w = (lsh - rsh).norm() + 1e-8
    return ((pts - c) @ B) / w


def setup(ax, title):
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8); ax.set_zlim(-2.2, 1.4)
    ax.set_box_aspect([1, 1, 1]); ax.view_init(elev=12, azim=-70); ax.axis("off")
    ax.set_title(title, fontsize=9)


fig = plt.figure(figsize=(18, 6))
for i in range(6):
    j = Js[i]; hc = canon(j, j[16], j[17])
    ax1 = fig.add_subplot(2, 6, i + 1, projection="3d")
    setup(ax1, "HUMAN  (3 joints/arm)" if i == 0 else "")
    for a, b, c in [(16, 18, 20), (17, 19, 21)]:
        arm = torch.stack([hc[a], hc[b], hc[c]]).numpy()
        ax1.plot(arm[:, 0], arm[:, 1], arm[:, 2], "-o", color="tab:blue", lw=3, ms=6)
    ax1.plot(*torch.stack([hc[16], hc[17]]).numpy().T, "-", color="gray", lw=2)

    # robot full chain
    lsh, rsh = rpos["arm_left_2_link"][i], rpos["arm_right_2_link"][i]
    ax2 = fig.add_subplot(2, 6, i + 7, projection="3d")
    setup(ax2, "ROBOT  (7 joints/arm)" if i == 0 else "")
    for chain in (LARM, RARM):
        pts = canon(torch.stack([rpos[n][i] for n in chain]), lsh, rsh).numpy()
        ax2.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", color="tab:red", lw=2.5, ms=5)
    ax2.plot(*canon(torch.stack([lsh, rsh]), lsh, rsh).numpy().T, "-", color="gray", lw=2)

plt.tight_layout()
out = media("retarget_full.png")
plt.savefig(out, dpi=95, bbox_inches="tight")
print("saved", out)
