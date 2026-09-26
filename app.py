import io
import base64

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torchvision.models as tv_models

from models.base_model import load_base_model, transform
from models.defense_model import load_defended_model
from models.attacks import get_attack

app = FastAPI(title="Adversarial Attack & Defense API")

# Loosen this to your real Lovable domain once you have it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load both models once at startup (not per-request) to keep latency low
base_model = load_base_model()
defended_model = load_defended_model()

# ImageNet class labels, pulled straight from torchvision's weights metadata
CLASSES = tv_models.ResNet18_Weights.DEFAULT.meta["categories"]


def predict(model, tensor):
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, idx = torch.max(probs, dim=1)
    return CLASSES[idx.item()], conf.item()


def image_to_base64(tensor):
    img = tensor.squeeze().permute(1, 2, 0).detach().numpy()
    img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    img = (img * 255).clip(0, 255).astype("uint8")
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "base_model": "ResNet18 (ImageNet pretrained)",
        "defended_model": "ResNet18 (adversarially trained via PGD)",
        "num_classes": len(CLASSES),
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    label, conf = predict(base_model, tensor)
    return {"prediction": label, "confidence": conf}


@app.post("/attack")
async def attack_endpoint(
    file: UploadFile = File(...),
    attack_type: str = Form("pgd"),
    epsilon: float = Form(0.03),
):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    label_clean, _ = predict(base_model, tensor)

    fake_label = torch.tensor([CLASSES.index(label_clean)])
    attack = get_attack(base_model, attack_type, epsilon)
    adv_tensor = attack(tensor, fake_label)

    label_adv, conf_adv = predict(base_model, adv_tensor)

    return {
        "original_prediction": label_clean,
        "adversarial_prediction": label_adv,
        "adversarial_confidence": conf_adv,
        "adversarial_image_base64": image_to_base64(adv_tensor),
        "attack_type": attack_type,
        "epsilon": epsilon,
    }


@app.post("/defend")
async def defend_endpoint(
    file: UploadFile = File(...),
    attack_type: str = Form("pgd"),
    epsilon: float = Form(0.03),
):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    label_clean, _ = predict(base_model, tensor)

    fake_label = torch.tensor([CLASSES.index(label_clean)])
    attack = get_attack(base_model, attack_type, epsilon)
    adv_tensor = attack(tensor, fake_label)

    label_def, conf_def = predict(defended_model, adv_tensor)

    return {"defended_prediction": label_def, "defended_confidence": conf_def}


@app.post("/compare")
async def compare_endpoint(
    file: UploadFile = File(...),
    attack_type: str = Form("pgd"),
    epsilon: float = Form(0.03),
):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    label_clean, conf_clean = predict(base_model, tensor)

    fake_label = torch.tensor([CLASSES.index(label_clean)])
    attack = get_attack(base_model, attack_type, epsilon)
    adv_tensor = attack(tensor, fake_label)

    label_adv, conf_adv = predict(base_model, adv_tensor)
    label_def, conf_def = predict(defended_model, adv_tensor)

    perturbation = (adv_tensor - tensor).squeeze().detach().numpy()
    perturbation_grid = np.linalg.norm(perturbation, axis=0).tolist()

    return {
        "original": {"prediction": label_clean, "confidence": conf_clean},
        "adversarial": {
            "prediction": label_adv,
            "confidence": conf_adv,
            "image_base64": image_to_base64(adv_tensor),
            "attack_type": attack_type,
            "epsilon": epsilon,
        },
        "defended": {"prediction": label_def, "confidence": conf_def},
        "perturbation_map": {"grid": perturbation_grid},
    }