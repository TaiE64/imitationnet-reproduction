"""
Visualize ImitationNet retargeting: human arm pose (top) vs retargeted TIAGo++ arm (bottom),
both expressed in the body frame (shoulder-centered, scaled to shoulder width) so the arm
CONFIGURATIONS are directly comparable. Like the paper's Fig.3.
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
_c = torch.load(ckpt("purefk"), map_location="cpu")
model = ImitationNet(latent=_c.get("latent", 8), hidden=_c.get("hidden", 128))
model.load_state_dict(_c["model"])
model.eval()

J = load_humanml3d_poses("test", max_motions=400, stride=15)   # (M,22,3)
# pick poses with diverse arm elevations (wrist height relative to shoulder)
elev = (J[:, 20, 2] + J[:, 21, 2]) / 2 - (J[:, 16, 2] + J[:, 17, 2]) / 2
order = torch.argsort(elev)
idx = order[torch.linspace(0, len(order) - 1, 6).long()]      # 6 from low->high arms
Js = J[idx]

Rh = human_limb_frames_canon(Js)
h_in = rotmat_to_6d(Rh).reshape(Rh.shape[0], -1)
with torch.no_grad():
    xr = fk.denormalize(model.retarget(h_in))                 # (6,14) robot joint angles
full = fk._full_config(xr)
fkres = fk.chain.forward_kinematics(full)
rpos = {k: v.get_matrix()[:, :3, 3] for k, v in fkres.items()}


def canon_pts(pts, lsh, rsh):
    """express world points (n,3) in body frame, shoulder-centered, unit shoulder-width."""
    B = body_frame(lsh, rsh)                                   # (3,3)
    c = (lsh + rsh) / 2
    w = (lsh - rsh).norm() + 1e-8
    return ((pts - c) @ B) / w                                 # B^T applied via right-mul


def draw(ax, sh_l, el_l, wr_l, sh_r, el_r, wr_r, color, title):
    for (a, b, c2) in [(sh_l, el_l, wr_l), (sh_r, el_r, wr_r)]:
        arm = torch.stack([a, b, c2]).numpy()
        ax.plot(arm[:, 0], arm[:, 1], arm[:, 2], "-o", color=color, lw=3, ms=6)
    sh = torch.stack([sh_l, sh_r]).numpy()
    ax.plot(sh[:, 0], sh[:, 1], sh[:, 2], "-", color="gray", lw=2)   # shoulders
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5); ax.set_zlim(-1.8, 1.2)
    ax.set_box_aspect([1, 1, 1]); ax.view_init(elev=12, azim=-70); ax.axis("off")


fig = plt.figure(figsize=(18, 6))
for i in range(6):
    # human arm points (canonicalized)
    j = Js[i]
    hc = canon_pts(j, j[16], j[17])
    ax1 = fig.add_subplot(2, 6, i + 1, projection="3d")
    draw(ax1, hc[16], hc[18], hc[20], hc[17], hc[19], hc[21], "tab:blue",
         "HUMAN" if i == 0 else "")
    # robot arm points (canonicalized): shoulder=arm_2, elbow=arm_4, wrist=arm_7
    L = lambda n: rpos[n][i]
    lsh, rsh = L("arm_left_2_link"), L("arm_right_2_link")
    pr = {n: canon_pts(torch.stack([L(n)]), lsh, rsh)[0] for n in
          ["arm_left_2_link", "arm_left_4_link", "arm_left_7_link",
           "arm_right_2_link", "arm_right_4_link", "arm_right_7_link"]}
    ax2 = fig.add_subplot(2, 6, i + 7, projection="3d")
    draw(ax2, pr["arm_left_2_link"], pr["arm_left_4_link"], pr["arm_left_7_link"],
         pr["arm_right_2_link"], pr["arm_right_4_link"], pr["arm_right_7_link"],
         "tab:red", "ROBOT (retargeted)" if i == 0 else "")

plt.tight_layout()
out = media("retarget_viz.png")
plt.savefig(out, dpi=90, bbox_inches="tight")
print("saved", out)
