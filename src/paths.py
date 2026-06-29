"""Project paths, resolved relative to this file so scripts work from any CWD."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (parent of src/)
CKPT = os.path.join(ROOT, "checkpoints")                            # trained models
MEDIA = os.path.join(ROOT, "media")                                # gifs / pngs


def ckpt(name):
    """checkpoint file for a run name, e.g. ckpt('purefk')."""
    return os.path.join(CKPT, name, "imitationnet.pt")


def media(fname):
    return os.path.join(MEDIA, fname)
