import os
from linecache import cache

import torch
import torchvision
import deep_danbooru_model
from PIL import Image
from pathlib import Path
from huggingface_hub import hf_hub_download
from pathlib import Path

device = "cpu"
torch.set_grad_enabled(False)
TORCH_DTYPE = torch.bfloat16

path_ = Path(os.path.abspath(__file__)).parent
cache_dir = path_ / "cache"

# https://huggingface.co/v2ray
# https://github.com/LagPixelLOL/deepgelbooru

def load_dgb_model(repo_id: str, filename: str, local_dir: str):
    try:
        os.makedirs(local_dir, exist_ok=True)
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, cache_dir=cache_dir)
        model_path = str(Path(local_dir, filename))
        model = deep_danbooru_model.DeepDanbooruModel.from_single_file(model_path, "cpu", TORCH_DTYPE)
        return model
    except Exception as e:
        print(e)
        return None


dgb_model = load_dgb_model("v2ray/deepgelbooru", "model_epoch_13.bin", "model")


def predict_deepgelbooru(image: Image.Image, threshold: float = 0.3):
    if image.mode != "RGB": image = image.convert("RGB")
    pic = image.resize((512, 512))
    x = torchvision.transforms.functional.pil_to_tensor(pic).to(device, TORCH_DTYPE).permute(1, 2, 0).unsqueeze(0) / 255

    dgb_model.to(device)
    r = dgb_model(x)
    dgb_model.to("cpu")
    y = r[0]

    tags = []
    for i, prob in sorted(((i, float(prob)) for i, prob in enumerate(y)), key=lambda x: x[1]):
        # print(model.tags[i], "-", prob)
        if prob > threshold: tags.append(dgb_model.tags[i])
    return ", ".join(tags)


def predict_tags_dgb(image: Image.Image, threshold: float = 0.3):
    def to_list(s):
        return [x.strip() for x in s.split(",") if not s == ""]

    tag_list = to_list(predict_deepgelbooru(image, threshold) + ", ")
    tag_list.remove("")
    return tag_list
