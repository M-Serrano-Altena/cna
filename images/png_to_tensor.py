from PIL import Image
import numpy as np
import torch

# Load the PNG, convert to float tensor [1, H, W]
def png_to_tensor(path: str):
    img = Image.open(path).convert('L')
    arr = np.array(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]

    return tensor