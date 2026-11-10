"""ricky (replacement for ringpole): ricky0526/so101_pick_toy_to_plate_v4 (v2.1 files, new-style calibration names).
White SO-101 on a dark mat in front of a white wall; 'top' = the dataset's 'front' camera (low, facing the arm),
'side' = the dataset's 'side' camera. Real masks: Grounding DINO ('a white robot arm.') + SAM 2.1 on every frame,
independent of the twin (segment_arm.py without --prompts)."""
import _v3joints


RED = [((235, 255), (90, 255), (60, 255)), ((0, 12), (90, 255), (60, 255))]   # pink base clamps, red block


def config() -> dict:
    # the cameras moved between recording sessions (phase correlation of each episode's first frame against
    # episode 0's static background, never the arm): front shifts after episode 4 and again from episode 13 on;
    # side shifts after episode 4. One pose per session.
    top0 = [0.0, 0.75, 0.15, -90.0, -5.0, 0.0, 70.0]
    side0 = [0.6, 0.5, 0.45, -150.0, -30.0, 0.0, 70.0]
    sess_top = dict(top=range(0, 5), top_b=range(5, 13), top_c=range(13, 80))
    sess_side = dict(side=range(0, 5), side_b=range(5, 80))
    views = {}
    for v, eps in sess_top.items():
        views[v] = dict(cam="top", init=top0, mount="world", role="arm", subtract_hsv=RED, episodes=list(eps))
    for v, eps in sess_side.items():
        views[v] = dict(cam="side", init=side0, mount="world", role="arm", subtract_hsv=RED, episodes=list(eps))
    return dict(real="so101_ricky_real", views=views, base_pos=(0.0, 0.0, 0.0), base_quat="0 0 0 1",
                real_to_sim=_v3joints.real_to_sim, gain_limit=[0.35, 0.35, 0.35, 0.35, 1.0, 0.0],   # -100..100 normalised units; roll may span ~a full turn
                real_mask="sam", parts=dict(servos=True),
                look=dict(body="0.95 0.95 0.94 1", servo="0.08 0.08 0.08 1"),
                scene=dict(table_rgba="0.10 0.10 0.11 1", wall_rgba="0.88 0.88 0.87 1"))
