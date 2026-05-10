# type: ignore

print("=" * 60)
print("TESTING IMPORTS")
print("=" * 60)


def test_import(statement: str):
    try:
        exec(statement, globals())
        print(f"✔️ SUCCESS: {statement}")
    except Exception as e:
        print(f"❌ FAILED: {statement}")
        print(f"   → {type(e).__name__}: {e}")


imports = [

    # Standard library
    "import os",
    "import json",
    "from pathlib import Path",

    # PyTorch ecosystem
    "import torch",
    "import torchvision",
    "import torchaudio",

    # Lightning
    "import lightning as L",

    # Logging / utilities
    "import wandb",
    "from tqdm import tqdm",

    # Plotting
    "import matplotlib",
    "import matplotlib.pyplot as plt",

    # Data handling
    "import pandas as pd",

    # Image processing
    "import cv2",
    "from skimage.draw import circle_perimeter, line",

    # Diff tools
    "from deepdiff import DeepDiff",

    # SSH tools
    "import paramiko",
]

for imp in imports:
    test_import(imp)

print("\n")
print("=" * 60)
print("TESTING PYTORCH")
print("=" * 60)

try:

    import torch

    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.tensor([1.0, 2.0], device=device)
    y = torch.tensor([3.0, 4.0], device=device)

    z = x + y

    print("Device:", device)
    print("x + y =", z)

    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))

    print("✔️ PyTorch test successful")

except Exception as e:

    print("❌ PyTorch test failed")
    print(f"   → {type(e).__name__}: {e}")

print("\n")
print("=" * 60)
print("TESTING TORCHVISION")
print("=" * 60)

try:

    import torchvision
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    print("Torchvision version:", torchvision.__version__)
    print("✔️ Torchvision test successful")

except Exception as e:

    print("❌ Torchvision test failed")
    print(f"   → {type(e).__name__}: {e}")

print("\n")
print("=" * 60)
print("TESTING LIGHTNING")
print("=" * 60)

try:

    import lightning as L
    import torch.nn as nn

    class DummyModel(L.LightningModule):

        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(2, 2)

        def forward(self, x):
            return self.layer(x)

    model = DummyModel()

    x = torch.randn(4, 2)

    y = model(x)

    print("Model output shape:", y.shape)

    print("✔️ Lightning test successful")

except Exception as e:

    print("❌ Lightning test failed")
    print(f"   → {type(e).__name__}: {e}")

print("\n")
print("=" * 60)
print("TESTING OpenCV")
print("=" * 60)

try:

    import cv2
    import numpy as np

    image = np.zeros((100, 100, 3), dtype=np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("Converted image shape:", gray.shape)

    print("✔️ OpenCV test successful")

except Exception as e:

    print("❌ OpenCV test failed")
    print(f"   → {type(e).__name__}: {e}")

print("\n")
print("=" * 60)
print("TESTING SKIMAGE")
print("=" * 60)

try:

    import skimage
    from skimage.draw import circle_perimeter, line
    import numpy as np

    img = np.zeros((64, 64), dtype=np.uint8)
    rr, cc = circle_perimeter(32, 32, 20)
    img[rr, cc] = 1
    rr, cc = line(0, 0, 63, 63)
    img[rr, cc] = 1

    print("skimage version:", skimage.__version__)
    print("circle_perimeter and line draw successful")
    print("✔️ skimage test successful")

except Exception as e:

    print("❌ skimage test failed")
    print(f"   → {type(e).__name__}: {e}")

print("\n")
print("=" * 60)
print("TESTING PANDAS")
print("=" * 60)

try:

    import pandas as pd

    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6]
    })

    print(df)

    print("✔️ Pandas test successful")

except Exception as e:

    print("❌ Pandas test failed")
    print(f"   → {type(e).__name__}: {e}")

print("\n")
print("=" * 60)
print("TESTING WANDB")
print("=" * 60)

try:

    import wandb

    print("wandb version:", wandb.__version__)

    print("✔️ wandb import successful")

except Exception as e:

    print("❌ wandb test failed")
    print(f"   → {type(e).__name__}: {e}")

print("\n")
print("=" * 60)
print("TESTING PARAMIKO")
print("=" * 60)

try:

    import paramiko

    client = paramiko.SSHClient()

    print("Paramiko client created successfully")

    print("✔️ Paramiko test successful")

except Exception as e:

    print("❌ Paramiko test failed")
    print(f"   → {type(e).__name__}: {e}")

print("\n")
print("=" * 60)
print("DONE")
print("=" * 60)