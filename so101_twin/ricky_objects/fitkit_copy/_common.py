"""Helpers for scene_<name>.py configs that reuse a builder's build_scene.CAMS / BASE_* and twin.real_to_sim."""
import pathlib

from armfit import load_module

TWIN = pathlib.Path(__file__).resolve().parents[2]   # objects copy


def builder(scene: str):
    d = TWIN / scene
    bs = load_module(d / "build_scene.py", f"{scene}_build_scene")
    tw = load_module(d / "twin.py", f"{scene}_twin")
    return bs, tw


def view(bs, cam: str, mount: str = "world", role: str = "arm", episodes=None, clean=None, set_cam=None, **kw) -> dict:
    c = bs.CAMS[cam]
    return dict(cam=set_cam or cam, init=[*c["pos"], c["yaw"], c["pitch"], c["roll"], c["fovy"]], mount=mount, role=role,
                episodes=episodes, clean=clean or {}, **kw)
