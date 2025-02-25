from .mlp import MLP
from .positional_encoding import PositionalEncodingsFixed

import torch
from torch import nn

from torchvision.ops import roi_align

class iEFLModule(nn.Module):
    def __init__(
        self,
        num_iterative_steps: int,
        emb_dim: int,
        kernel_dim: int,
        num_objects: int,
        num_heads: int,
        reduction: int,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
        zero_shot: bool,
    ):
        super().__init__()

        self.num_iterative_steps = num_iterative_steps
        self.zero_shot = zero_shot
        self.kernel_dim = kernel_dim
        self.num_objects = num_objects
        self.emb_dim = emb_dim
        self.reduction = reduction

        # Shape mapping network (MLP untuk shape embedding)
        self.shape_mapping = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, kernel_dim * kernel_dim * emb_dim)
        )

        # Multi-head cross attention untuk fusi feature
        self.exemplar_attention = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=0.0, batch_first=True
        )
        
        # Linear attention untuk image feature interaction
        self.linear_attention = LinearAttention(emb_dim)
        
        # Feature feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(emb_dim, mlp_factor * emb_dim),
            activation(),
            nn.Linear(mlp_factor * emb_dim, emb_dim)
        )
        
        # Layer normalizations
        self.norm1 = nn.LayerNorm(emb_dim) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(emb_dim) if norm else nn.Identity()
        self.norm3 = nn.LayerNorm(emb_dim) if norm else nn.Identity()

        # Positional encoding
        self.pos_emb = PositionalEncodingsFixed(emb_dim)

        if self.zero_shot:
            self.shape_or_objectness = nn.Parameter(
                torch.empty((self.num_objects, self.kernel_dim**2, emb_dim))
            )
            nn.init.normal_(self.shape_or_objectness)

    def forward(self, f_e, pos_emb, bboxes):
        bs, _, h, w = f_e.size()
        
        # Extract shape information dari bounding boxes
        if not self.zero_shot:
            box_hw = torch.zeros(bboxes.size(0), bboxes.size(1), 2).to(bboxes.device)
            box_hw[:, :, 0] = bboxes[:, :, 2] - bboxes[:, :, 0]  # width
            box_hw[:, :, 1] = bboxes[:, :, 3] - bboxes[:, :, 1]  # height
            shape_emb = self.shape_mapping(box_hw).reshape(
                bs, -1, self.kernel_dim ** 2, self.emb_dim
            ).flatten(1, 2).transpose(0, 1)
        else:
            shape_emb = self.shape_or_objectness.expand(
                bs, -1, -1, -1
            ).flatten(1, 2).transpose(0, 1)

        # Extract exemplar features menggunakan RoIAlign jika tidak zero-shot
        if not self.zero_shot:
            boxes_flat = torch.cat([
                torch.arange(bs, device=bboxes.device).repeat_interleave(
                    self.num_objects
                ).reshape(-1, 1),
                bboxes.flatten(0, 1),
            ], dim=1)
            exemplar_features = roi_align(
                f_e,
                boxes=boxes_flat, 
                output_size=self.kernel_dim,
                spatial_scale=1.0/self.reduction, 
                aligned=True
            ).permute(0, 2, 3, 1).reshape(
                bs, self.num_objects * self.kernel_dim ** 2, -1
            ).transpose(0, 1)
        else:
            exemplar_features = None

        # Generate positional encodings
        query_pos_emb = self.pos_emb(
            bs, self.kernel_dim, self.kernel_dim, f_e.device
        ).flatten(2).permute(2, 0, 1).repeat(self.num_objects, 1, 1)

        # Iterative feature learning
        all_prototypes = []
        F_k = exemplar_features if exemplar_features is not None else shape_emb

        memory = f_e.flatten(2).permute(2, 0, 1)

        for k in range(self.num_iterative_steps):
            # 1. Fuse shape and appearance dengan MHCA
            if not self.zero_shot:
                F_k = self.norm1(F_k + self.exemplar_attention(
                    query=F_k + query_pos_emb,
                    key=shape_emb,
                    value=shape_emb
                )[0])
            
            # 2. Learn dari image features dengan linear attention
            F_k = self.norm2(F_k + self.linear_attention(
                query=F_k + query_pos_emb,
                key_value=memory + pos_emb
            ))
            
            # 3. Feed-forward transformation
            F_k = self.norm3(F_k + self.ff_network(F_k))
            
            all_prototypes.append(F_k)

        # Stack semua prototype features
        return torch.stack(all_prototypes)


class LinearAttention(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5
        
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        
    def forward(self, query, key_value):
        q = self.q_proj(query) * self.scale
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)
        
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        
        context = torch.matmul(k.transpose(-2, -1), v)
        out = torch.matmul(q, context)
        
        return out