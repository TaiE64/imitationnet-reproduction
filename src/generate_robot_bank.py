"""
Generate a large robot-pose bank (paper uses ~15M TIAGo++ poses sampled from the
configuration space + FK). Chunked FK to bound memory; cached to disk so training
doesn't regenerate it.

Saves: poses (N,14) normalized to [-1,1] + limb dirs (N,4,3) -> robot_bank.npz
"""
import argparse
import time
import numpy as np
import torch
from robot_kinematics import RobotFK
from rotations import rotmat_to_6d

OUT = "/home/taie/Downloads/Reproduce/datasets/robot_retargeting/robot_bank.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15_000_000)   # paper: ~15M
    ap.add_argument("--chunk", type=int, default=250_000)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    fk = RobotFK(device="cpu")
    poses, rots = [], []
    done, t0 = 0, time.time()
    while done < args.n:
        b = min(args.chunk, args.n - done)
        x = fk.sample_arm_poses(b)              # (b,14) radians
        R = fk.limb_frames_canon(x)             # (b,4,3,3) body-canonical limb rotations
        r6 = rotmat_to_6d(R).reshape(b, -1)     # (b,24) 6D for compact storage
        xn = fk.normalize(x)                    # (b,14) [-1,1]
        poses.append(xn.numpy().astype(np.float32))
        rots.append(r6.numpy().astype(np.float32))
        done += b
        if done % 1_000_000 == 0 or done == args.n:
            print(f"  {done/1e6:.1f}M / {args.n/1e6:.0f}M  ({done/(time.time()-t0):.0f} poses/s)")
    poses = np.concatenate(poses, 0)
    rots = np.concatenate(rots, 0)             # (N,24)
    np.savez(args.out, poses=poses, rots=rots)
    print(f"saved {poses.shape[0]} poses -> {args.out}  ({(poses.nbytes + rots.nbytes)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
