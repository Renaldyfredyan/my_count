from backbone_swin import Backbone
from hybrid_encoder import HybridEncoder
from ielf import iEFLModule
from positional_encoding import PositionalEncodingsFixed
from regression_head import DensityMapRegressor

import torch
from torch import nn
from torch.nn import functional as F


class efficient(nn.Module):

    def __init__(
        self,
        image_size: int,
        num_encoder_layers: int,
        num_iefl_iterative_steps: int,
        num_objects: int,
        emb_dim: int,
        num_heads: int,
        kernel_dim: int,
        backbone_name: str,
        train_backbone: bool,
        reduction: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
        zero_shot: bool,
    ):

        super(efficient, self).__init__()

        self.emb_dim = emb_dim
        self.num_objects = num_objects
        self.reduction = reduction
        self.kernel_dim = kernel_dim
        self.image_size = image_size
        self.zero_shot = zero_shot
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers

        self.backbone = Backbone(
            pretrained=True,
            reduction=reduction,
            requires_grad=train_backbone
        )

        # self.input_proj = nn.Conv2d(
        #     self.backbone.total_channels, emb_dim, kernel_size=1
        # )

        if num_encoder_layers > 0:
            self.encoder = HybridEncoder(
                num_encoder_layers, emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation, norm
            )
        else:
            self.input_proj = nn.Conv2d(self.backbone.total_channels, emb_dim, kernel_size=1)

        self.iefl = iEFLModule(
            num_iterative_steps=num_iefl_iterative_steps,
            emb_dim=emb_dim,
            kernel_dim=kernel_dim,
            num_objects=num_objects,
            num_heads=num_heads,
            dropout=dropout,
            reduction=reduction,
            layer_norm_eps=layer_norm_eps,
            mlp_factor=mlp_factor,
            norm_first=norm_first,
            activation=activation,
            norm=norm,
            zero_shot=zero_shot
        )

        self.regression_head = DensityMapRegressor(emb_dim, reduction)
        self.aux_heads = nn.ModuleList([
            DensityMapRegressor(emb_dim, reduction)
            for _ in range(num_iefl_iterative_steps - 1)
        ])

        self.pos_emb = PositionalEncodingsFixed(emb_dim)

    def forward(self, x, bboxes):
  
        num_objects = bboxes.size(1) if not self.zero_shot else self.num_objects
        
        # Get backbone features
        s3, s4, s5 = self.backbone.forward_multiscale(x)  # Ambil multi-scale features
        
        # Jika ada encoder, lakukan hybrid encoding
        if hasattr(self, 'encoder'):
            image_features = self.encoder(s3, s4, s5)
                                        #   None)
        else:
            # Jika tidak ada encoder, concat dan project features
            backbone_features = self.backbone.forward_concatenated(x)
            image_features = self.input_proj(backbone_features)

        # prepare iefl input
        bs, c, h, w = image_features.size()
        f_e = image_features
        
        # Generate positional embeddings dengan ukuran yang sesuai
        pos_emb = self.pos_emb(
            bs, h, w,  # Gunakan h, w dari image_features
            x.device
        ).flatten(2).permute(2, 0, 1)
        
        # # Print untuk debug
        # print("f_e shape:", f_e.shape)
        # print("pos_emb shape:", pos_emb.shape)
        # print("memory shape after permute:", f_e.flatten(2).permute(2, 0, 1).shape)

        all_prototypes = self.iefl(f_e, pos_emb, bboxes)
        
        outputs = list()


        for i in range(all_prototypes.size(0)):
            prototypes = all_prototypes[i, ...].permute(1, 0, 2).reshape(
                bs, num_objects, self.kernel_dim, self.kernel_dim, -1
            ).permute(0, 1, 4, 2, 3).flatten(0, 2)[:, None, ...]

            response_maps = F.conv2d(
                torch.cat([f_e for _ in range(num_objects)], dim=1).flatten(0, 1).unsqueeze(0),
                prototypes,
                bias=None,
                padding=self.kernel_dim // 2,
                groups=prototypes.size(0)
            ).view(
                bs, num_objects, self.emb_dim, h, w
            ).max(dim=1)[0]

            # send through regression heads
            if i == all_prototypes.size(0) - 1:
                predicted_dmaps = self.regression_head(response_maps)
            else:
                predicted_dmaps = self.aux_heads[i](response_maps)
            outputs.append(predicted_dmaps)

        return outputs[-1], outputs[:-1]


def build_model(args):

    assert args.backbone in ['grounding_dino']
    assert args.reduction in [4, 8, 16]

    return efficient(
        image_size=args.image_size,
        num_encoder_layers=args.num_enc_layers,
        num_iefl_iterative_steps=args.num_iefl_iterative_steps,
        num_objects=args.num_objects,
        zero_shot=args.zero_shot,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        kernel_dim=args.kernel_dim,
        backbone_name=args.backbone,
        train_backbone=args.backbone_lr > 0,
        reduction=args.reduction,
        dropout=args.dropout,
        layer_norm_eps=1e-5,
        mlp_factor=8,
        norm_first=args.pre_norm,
        activation=nn.GELU,
        norm=True,
    )
