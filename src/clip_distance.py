import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def load_clip(model_name: str = "openai/clip-vit-base-patch32"):
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def compute_distance_matrix(
    tokens: list[str],
    image: Image.Image,
    model: CLIPModel = None,
    processor: CLIPProcessor = None,
) -> np.ndarray:
    """
    Returns distance matrix of shape [num_text_tokens, 16] where each value
    is 1 - cosine_similarity between a text token embedding and a patch embedding.
    """
    if model is None or processor is None:
        model, processor = load_clip()

    # Split image into 4x4 grid of 16 patches
    w, h = image.size
    pw, ph = w // 4, h // 4
    patches = []
    for row in range(4):
        for col in range(4):
            patch = image.crop((col * pw, row * ph, (col + 1) * pw, (row + 1) * ph))
            patches.append(patch)

    # Compute patch embeddings
    patch_inputs = processor(images=patches, return_tensors="pt", padding=True)
    with torch.no_grad():
        patch_embeds = model.get_image_features(**patch_inputs)
    patch_embeds = patch_embeds / patch_embeds.norm(dim=-1, keepdim=True)  # [16, D]

    # Compute text token embeddings
    text_inputs = processor(text=tokens, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)  # [T, D]

    # Cosine similarity: [T, 16], distance = 1 - sim
    cosine_sim = torch.matmul(text_embeds, patch_embeds.T)  # [T, 16]
    distance_matrix = 1.0 - cosine_sim.numpy()

    return distance_matrix


if __name__ == "__main__":
    from PIL import Image, ImageDraw

    # Create a simple test image
    img = Image.new("RGB", (224, 224), color=(128, 64, 200))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 174, 174], fill=(255, 128, 0))

    tokens = ["dog", "cat", "sky", "orange square"]

    model, processor = load_clip()
    dist = compute_distance_matrix(tokens, img, model, processor)

    print(f"Distance matrix shape: {dist.shape}")
    print(dist)
