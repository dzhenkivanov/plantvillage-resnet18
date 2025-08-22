import io, torch
from PIL import Image
from torchvision.transforms import v2
from torchvision.models import resnet18
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_eval_transform(size=128):
    return v2.Compose([v2.Resize(size), v2.ToTensor(), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

def load_model():
    import json
    with open("app/labels.json") as f:
        class_to_idx = json.load(f)
    class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    state = torch.load("app/plantvillage.pt", map_location="cpu")
    state = state.get("model_state", state)
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    transform = build_eval_transform(128)
    return model, class_names, transform

def predict_image(image_bytes, model, transform, class_names, k=3):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        topk = torch.topk(probs, k)
    return [{"label": class_names[i], "prob": float(p)} for p, i in zip(topk.values, topk.indices)]
