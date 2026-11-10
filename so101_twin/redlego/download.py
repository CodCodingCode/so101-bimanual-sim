import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id="eruedas/so101_3cams_pickplace-red-lego_real",
    repo_type="dataset",
    local_dir="data_so101/eruedas__so101_3cams_pickplace-red-lego_real",
    allow_patterns=["meta/**", "data/**", "videos/observation.images.top/**", "videos/observation.images.left/**"],
    max_workers=8,
)
print("done", p)
