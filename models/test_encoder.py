import torch
from encoder import HHFSAMEncoder

# Load our encoder
model = HHFSAMEncoder(
    checkpoint_path="checkpoints/sam_vit_b_01ec64.pth"
)

# Create a fake RGB image
x = torch.randn(1, 3, 1024, 1024)

# Run the image through the encoder
with torch.no_grad():
    features, neck_output = model(x)

print("Encoder ran successfully!")

for i, feature in enumerate(features):
    print(f"Feature {i + 1} shape:", feature.shape)

print("Neck output shape:", neck_output.shape)
