import os
import torch
import torchvision.models as models

WEIGHTS_PATH = "weights/defended_model.pth"

def load_defended_model():
    model = models.resnet18(weights=None)
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
        print("Loaded adversarially trained weights.")
    else:
        print("WARNING: defended_model.pth not found. Using pretrained "
              "ImageNet weights as a placeholder — /defend will not show "
              "real robustness yet.")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    return model