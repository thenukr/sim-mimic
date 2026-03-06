import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

_MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"

def load_model(device=None):
    processor = AutoImageProcessor.from_pretrained(_MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(_MODEL_ID).eval()
    if device is not None:
        model = model.to(device)
    return processor, model

def get_features(image_path, processor, model, layer=-1):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[layer]  # (B, 1370, 1024)

"""

processor, model = load_model(device="cuda")

# in your preprocessing loop
features = get_features("frame_001.png", processor, model, layer=-1)
torch.save(features, "frame_001_features.pt")


"""