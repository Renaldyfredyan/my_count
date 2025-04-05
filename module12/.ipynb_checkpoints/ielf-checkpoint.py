from mlp import MLP
from positional_encoding import PositionalEncodingsFixed

import torch
from torch import nn

from torchvision.ops import roi_align

class LinearAttention(nn.Module):
    def __init__(self, emb_dim, dropout=0.1, norm_first=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.scale = emb_dim ** -0.5
        self.norm_first = norm_first
        
        self.q_proj = nn.Linear(emb_dim, emb_dim)
        self.k_proj = nn.Linear(emb_dim, emb_dim)
        self.v_proj = nn.Linear(emb_dim, emb_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Normalization layer for pre-norm variant
        self.norm = nn.LayerNorm(emb_dim)
        
    def forward(self, query, key_value, query_pos=None, key_pos=None):
        """
        Efficient linear attention implementation
        Args:
            query: [N*B, E] tensor of query features
            key_value: [HW*B, E] tensor of key/value features
            query_pos: Optional positional encoding for query
            key_pos: Optional positional encoding for key
        Returns:
            output: [N*B, E] tensor of updated query features
        """
        # Apply positional encodings if provided
        q = query
        if query_pos is not None:
            q = q + query_pos
            
        k = key_value
        if key_pos is not None:
            k = k + key_pos
        
        # Apply normalization if using pre-norm
        if self.norm_first:
            q = self.norm(q)
            k = self.norm(k)  # Pastikan norm digunakan juga untuk k
        
        # Linear projections
        q = self.q_proj(q) * self.scale  # [N*B, E]
        k = self.k_proj(k)  # [HW*B, E]
        v = self.v_proj(key_value)  # [HW*B, E]
        
        # Apply softmax along feature dimension
        q = q.softmax(dim=-1)  # [N*B, E]
        k = k.softmax(dim=0)    # [HW*B, E]
        
        # Linear attention computation
        # First compute context: k^T * v
        context = torch.einsum('be,be->e', k, v)  # [E]
        
        # Then compute output: q * context
        out = torch.einsum('be,e->be', q, context)  # [N*B, E]
        
        # Apply normalization after if using post-norm
        if not self.norm_first:
            out = self.norm(out)  # Pastikan norm digunakan juga dalam kasus post-norm
        
        return out

class iEFLModule(nn.Module):
    def __init__(
        self,
        num_iterative_steps: int,
        emb_dim: int,
        kernel_dim: int,
        num_objects: int,
        num_heads: int,
        dropout: float,
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
        self.norm_first = norm_first

        # Shape mapping network
        if not self.zero_shot:
            self.shape_mapping = nn.Sequential(
                nn.Linear(2, 64),
                nn.ReLU(),
                nn.Linear(64, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, kernel_dim * kernel_dim * emb_dim)
            )
        else:
            self.shape_mapping = nn.Parameter(
                torch.empty((self.num_objects, self.kernel_dim**2, emb_dim))
            )
            nn.init.normal_(self.shape_mapping)

        # Multi-head cross attention for exemplar and shape fusion (Step 1)
        self.exemplar_shape_attention = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout, batch_first=False
        )
        
        # Multi-head cross attention for exemplar-image interaction (Step 2)
        self.exemplar_image_attention = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=dropout, batch_first=False
        )
        
        # Linear attention for the next part of the pipeline (as shown in the figure)
        self.linear_attention = LinearAttention(
            emb_dim, 
            dropout=dropout,
            norm_first=norm_first
        )
        
        # Feed-forward network (Step 3)
        self.ff_network = MLP(emb_dim, mlp_factor * emb_dim, dropout, activation)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(emb_dim, eps=layer_norm_eps) if norm else nn.Identity()
        self.norm2 = nn.LayerNorm(emb_dim, eps=layer_norm_eps) if norm else nn.Identity()
        self.norm3 = nn.LayerNorm(emb_dim, eps=layer_norm_eps) if norm else nn.Identity()
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Positional encoding
        self.pos_emb = PositionalEncodingsFixed(emb_dim)

        # Flag untuk menggunakan MHCA atau Linear Attention di step 2
        self.use_mhca_for_step2 = True  # Set True untuk mengikuti paper

    def forward(self, f_e, pos_emb, bboxes):
        bs, _, h, w = f_e.size()
        
        # Extract shape information from bounding boxes
        if not self.zero_shot:
            box_hw = torch.zeros(bboxes.size(0), bboxes.size(1), 2).to(bboxes.device)
            box_hw[:, :, 0] = bboxes[:, :, 2] - bboxes[:, :, 0]  # width
            box_hw[:, :, 1] = bboxes[:, :, 3] - bboxes[:, :, 1]  # height
            shape_emb = self.shape_mapping(box_hw).reshape(
                bs, -1, self.kernel_dim ** 2, self.emb_dim
            ).flatten(1, 2).transpose(0, 1)  # [N, B, E]
        else:
            shape_emb = self.shape_mapping.expand(
                bs, -1, -1, -1
            ).flatten(1, 2).transpose(0, 1)  # [N, B, E]

        # Extract exemplar features using RoIAlign if not zero-shot
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
            ).transpose(0, 1).contiguous()  # [N, B, E]
        else:
            # Dalam mode zero-shot, gunakan shape_emb sebagai exemplar_features
            exemplar_features = shape_emb.clone()

        # Generate positional encodings for queries
        query_pos_emb = self.pos_emb(
            bs, self.kernel_dim, self.kernel_dim, f_e.device
        ).flatten(2).permute(2, 0, 1).repeat(self.num_objects, 1, 1).contiguous()  # [N, B, E]

        # Prepare memory (image features)
        memory = f_e.flatten(2).permute(2, 0, 1).contiguous()  # [HW, B, E]

        # Iterative feature learning
        all_prototypes = []
        
        # Initialize F_S_exm dengan shape_emb (sesuai Algorithm 1)
        F_S_exm = shape_emb
        
        # Initialize F_exm dengan exemplar_features (sesuai Algorithm 1)
        F_exm = exemplar_features
        
        for k in range(self.num_iterative_steps):
            # Step 1: MHCA for fusing shape and exemplar
            if self.norm_first:
                F_S_norm = self.norm1(F_S_exm)
                attn_output = self.exemplar_shape_attention(
                    query=F_S_norm + query_pos_emb,
                    key=F_exm,
                    value=F_exm
                )[0]
                F_prime_exm = F_S_exm + self.dropout1(attn_output)
            else:
                attn_output = self.exemplar_shape_attention(
                    query=F_S_exm + query_pos_emb,
                    key=F_exm,
                    value=F_exm
                )[0]
                F_prime_exm = self.norm1(F_S_exm + self.dropout1(attn_output))
            
            # Step 2: First MHCA with image features
            if self.norm_first:
                F_prime_norm = self.norm2(F_prime_exm)
                attn_output = self.exemplar_image_attention(
                    query=F_prime_norm,
                    key=memory,
                    value=memory
                )[0]
                F_hat_k_mhca = F_prime_exm + self.dropout2(attn_output)
            else:
                attn_output = self.exemplar_image_attention(
                    query=F_prime_exm,
                    key=memory,
                    value=memory
                )[0]
                F_hat_k_mhca = self.norm2(F_prime_exm + self.dropout2(attn_output))
            
            # Then apply linear attention as shown in the figure
            F_prime_flat = F_hat_k_mhca.reshape(-1, self.emb_dim)
            memory_flat = memory.reshape(-1, self.emb_dim)
            
            # Apply linear attention - pastikan LayerNorm dalam LinearAttention digunakan
            attn_output = self.linear_attention(
                query=F_prime_flat,
                key_value=memory_flat,
                query_pos=query_pos_emb.reshape(-1, self.emb_dim),
                key_pos=pos_emb.reshape(-1, self.emb_dim) if pos_emb is not None else None
            )
            
            # Reshape back and continue
            F_hat_k = attn_output.reshape(F_hat_k_mhca.shape)
            
            # Step 3: Feed-forward transformation (F_S(k+1)_exm = FF(F^k) + F^k)
            if self.norm_first:
                # Pre-norm for Step 3
                F_hat_norm = self.norm3(F_hat_k)
                ff_output = self.ff_network(F_hat_norm)
                F_S_exm = F_hat_k + self.dropout3(ff_output)  # Update F_S_exm untuk iterasi berikutnya
            else:
                # Post-norm for Step 3
                ff_output = self.ff_network(F_hat_k)
                F_S_exm = self.norm3(F_hat_k + self.dropout3(ff_output))  # Update F_S_exm untuk iterasi berikutnya
            
            # Add to the list of all prototypes (F^k yang sudah diproses)
            all_prototypes.append(F_hat_k)

        # Stack all iterations of prototype features
        return torch.stack(all_prototypes)  # [T, N, B, E]