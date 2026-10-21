import torch
import torch.nn as nn
from segment_anything import sam_model_registry
class LoRALayer(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()

        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)

    def forward(self, x):
        return self.up(self.down(x))

class LoRAQKV(nn.Module):
    def __init__(self, original_qkv, dim, rank):
        super().__init__()

        self.original_qkv = original_qkv
        self.lora_q = LoRALayer(dim, rank)
        self.lora_v = LoRALayer(dim, rank)

    def forward(self, x):
        qkv = self.original_qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q + self.lora_q(x)
        v = v + self.lora_v(x)

        return torch.cat([q, k, v], dim=-1)

class Adapter(nn.Module):
    def __init__(self, dim, adapter_dim):
        super().__init__()

        self.down_project = nn.Linear(dim, adapter_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_dim, dim)

    def forward(self, x):
        x = self.down_project(x)
        x = self.activation(x)
        x = self.up_project(x)

        return x
class MedAdapterBlock(nn.Module):
    def __init__(self, original_block, dim, adapter_dim):
        super().__init__()

        self.original_block = original_block
        self.adapter = Adapter(dim, adapter_dim)

    def forward(self, x):
        # SAM attention part
        shortcut = x

        x = self.original_block.norm1(x)
        x = self.original_block.attn(x)
        x = shortcut + x

        # SAM MLP part
        mlp_input = self.original_block.norm2(x)
        mlp_output = self.original_block.mlp(mlp_input)

        # Medical Adapter
        adapter_output = self.adapter(mlp_output)

        # Residual fusion
        x = x + mlp_output + adapter_output

        return x

class HHFSAMEncoder(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()

        # Load pretrained SAM-B
        self.sam = sam_model_registry["vit_b"](
            checkpoint=checkpoint_path
        )

        # Freeze original SAM parameters
        for parameter in self.sam.parameters():
            parameter.requires_grad = False
          
                   # Add LoRA to Query and Value of every Transformer block
        for block in self.sam.image_encoder.blocks:
            dim = block.attn.qkv.in_features

            block.attn.qkv = LoRAQKV(
                original_qkv=block.attn.qkv,
                dim=dim,
                rank=4
            )

        # Add Adapter to every SAM Transformer block
        for i in range(len(self.sam.image_encoder.blocks)):
            block = self.sam.image_encoder.blocks[i]
            dim = block.attn.qkv.original_qkv.in_features

            self.sam.image_encoder.blocks[i] = MedAdapterBlock(
                original_block=block,
                dim=dim,
                adapter_dim=64
            )
            

class HHFSAMEncoder(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()

        # Load pretrained SAM-B
        self.sam = sam_model_registry["vit_b"](
            checkpoint=checkpoint_path
        )

        # Freeze original SAM parameters
        for parameter in self.sam.parameters():
            parameter.requires_grad = False

        # Add LoRA to Query and Value
        for block in self.sam.image_encoder.blocks:
            dim = block.attn.qkv.in_features

            block.attn.qkv = LoRAQKV(
                original_qkv=block.attn.qkv,
                dim=dim,
                rank=4
            )

        # Add Adapter to every Transformer block
        for i in range(len(self.sam.image_encoder.blocks)):
            block = self.sam.image_encoder.blocks[i]
            dim = block.attn.qkv.original_qkv.in_features

            self.sam.image_encoder.blocks[i] = MedAdapterBlock(
                original_block=block,
                dim=dim,
                adapter_dim=64
            )
    def forward(self, x):
        # Patch embedding
        x = self.sam.image_encoder.patch_embed(x)

        # Add positional embedding
        if self.sam.image_encoder.pos_embed is not None:
            x = x + self.sam.image_encoder.pos_embed

        features = []

        # Pass through all Transformer blocks   
        for i, block in enumerate(self.sam.image_encoder.blocks):
            x = block(x)

            # Save outputs of layers 3, 6, 9 and 12
            if i in [2, 5, 8, 11]:
                features.append(x)

        # Pass final feature through SAM neck
        neck_output = self.sam.image_encoder.neck(
            x.permute(0, 3, 1, 2)
        )

        return features, neck_output
