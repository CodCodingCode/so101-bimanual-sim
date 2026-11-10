"""ringpole: nbirukov/so101_pick_stack_ring_pole. White arm, black servos, dark wood table; fixed overhead 'top'.
The 'side' camera rides on the shoulder and looks at the pole (the arm is almost never in it): not scored."""
from _common import builder, view


def config() -> dict:
    bs, tw = builder("ringpole")
    views = {"top": view(bs, "top", clean=dict(link=8, keep="largest", min_px=300),
                         exclude=[(0, 22, 0, 640)],   # white wall strip along the top edge
                         # orange ring / purple pole base: SAM sometimes glues the held ring to the arm
                         subtract_hsv=[((0, 30), (110, 255), (70, 255)), ((240, 255), (110, 255), (70, 255)),
                                       ((180, 220), (50, 255), (60, 255))])}
    return dict(real="so101_ringpole_real", views=views, base_pos=bs.BASE_POS, base_quat=bs.BASE_QUAT,
                real_to_sim=tw.real_to_sim, real_mask="auto", wrist_cam=True,
                look=dict(body="0.93 0.92 0.90 1", servo="0.08 0.08 0.08 1", jaw="0.93 0.92 0.90 1"))
