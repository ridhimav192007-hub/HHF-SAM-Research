import torch
from mhpm import MHPM

# Create a sample encoder feature
x = torch.randn(1, 768, 64, 64)

# Initialize MHPM
model = MHPM(in_channels=768, out_channels=256)

# Run MHPM
model.eval()

with torch.no_grad():
    output = model(x)

print("MHPM ran successfully!")
print("Input shape:", x.shape)
print("Output shape:", output.shape)
