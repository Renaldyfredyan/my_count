from .backbone import Backbone
from .transformer import TransformerEncoder
from .ope import OPEModule
from .positional_encoding import PositionalEncodingsFixed
from .regression_head import DensityMapRegressor
from .selfattention import SelfAttention
from .crossscale import CrossScaleFusion
from .iefl import IterativeExemplarFeatureLearning

import torch
from torch import nn
from torch.nn import functional as F
import math


class train_one_epoch(nn.Module):

    def __init__(
        self,
        image_size: int,
        num_encoder_layers: int,
        num_ope_iterative_steps: int,
        num_objects: int,
        emb_dim: int,
        num_heads: int,
        kernel_dim: int,
        backbone_name: str,
        swav_backbone: bool,
        train_backbone: bool,
        reduction: int,
        dropout: float,
        layer_norm_eps: float,
        mlp_factor: int,
        norm_first: bool,
        activation: nn.Module,
        norm: bool,
        zero_shot: bool,
        num_iterations: int,
        
    ):

        super(train_one_epoch, self).__init__()

        self.emb_dim = emb_dim
        self.num_objects = num_objects
        self.reduction = reduction
        self.kernel_dim = kernel_dim
        self.image_size = image_size
        self.zero_shot = zero_shot
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers

        self.self_attn = SelfAttention(emb_dim, emb_dim, num_heads)

        self.fusion1 = CrossScaleFusion(emb_dim, emb_dim)
        self.fusion2 = CrossScaleFusion(emb_dim, emb_dim)

        self.i_efl = IterativeExemplarFeatureLearning(emb_dim, num_heads, num_iterations)

        self.backbone = Backbone(
            backbone_name, pretrained=True, dilation=False, reduction=reduction,
            swav=swav_backbone, requires_grad=train_backbone
        )
        self.input_proj = nn.Conv2d(
            self.backbone.num_channels, emb_dim, kernel_size=1
        )

        if num_encoder_layers > 0:
            self.encoder = TransformerEncoder(
                num_encoder_layers, emb_dim, num_heads, dropout, layer_norm_eps,
                mlp_factor, norm_first, activation, norm
            )

        self.ope = OPEModule(
            num_ope_iterative_steps, emb_dim, kernel_dim, num_objects, num_heads,
            reduction, layer_norm_eps, mlp_factor, norm_first, activation, norm, zero_shot
        )

        self.regression_head = DensityMapRegressor(emb_dim, reduction)
        self.aux_heads = nn.ModuleList([
            DensityMapRegressor(emb_dim, reduction)
            for _ in range(num_ope_iterative_steps - 1)
        ])

        self.pos_emb = PositionalEncodingsFixed(emb_dim)

    def forward(self, x, bboxes):
        num_objects = bboxes.size(1) if not self.zero_shot else self.num_objects
        
        # backbone
        backbone_features = self.backbone(x)

        # prepare the encoder input
        src = self.input_proj(backbone_features)
        bs, c, h, w = src.size()
        pos_emb = self.pos_emb(bs, h, w, src.device).flatten(2).permute(2, 0, 1)
        
        # Ubah src menjadi format yang sesuai untuk self-attention
        src = src.flatten(2).permute(2, 0, 1)  # [h*w, batch, channel]

        # Inisialisasi features dengan shape yang benar
        S4 = src
        S5 = src
        image_features = src
        exemplar_features = src

        # Pastikan S5 memiliki shape yang benar untuk self-attention
        S5 = self.self_attn(S5)
        B = S5.size(1)  # batch size
        H = W = int(math.sqrt(S5.size(0)))  # assuming square feature maps
        S5 = S5.permute(1, 2, 0).view(B, self.emb_dim, H, W)
        S4 = S4.permute(1, 2, 0).view(B, self.emb_dim, H, W)

        S4 = self.fusion1(S4, S5)
        S3 = self.fusion2(S4, S5)

        # Iterative Exemplar Feature Learning
        exemplar_features = self.i_efl(exemplar_features, image_features)

        # push through the encoder
        if self.num_encoder_layers > 0:
            image_features = self.encoder(src, pos_emb, src_key_padding_mask=None, src_mask=None)
        else:
            image_features = src

        # prepare OPE input
        f_e = image_features.permute(1, 2, 0).reshape(-1, self.emb_dim, h, w)

        all_prototypes = self.ope(f_e, pos_emb, bboxes)

        exemplar_features = exemplar_features.permute(1, 2, 0).reshape(bs, self.emb_dim, h, w)
        exemplar_features = F.interpolate(exemplar_features, 
                                    size=(self.image_size, self.image_size),
                                    mode='bilinear', 
                                    align_corners=False)


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
                
            # Resize ke ukuran input original
            predicted_dmaps = F.interpolate(predicted_dmaps, 
                                        size=(self.image_size, self.image_size),
                                        mode='bilinear', 
                                        align_corners=False)
            outputs.append(predicted_dmaps)

        return outputs[-1], outputs[:-1], exemplar_features
def build_model(args):

    assert args.backbone in ['resnet18', 'resnet34', 'resnet50', 'swinT1k']
    assert args.reduction in [4, 8, 16]

    return train_one_epoch(
        image_size=args.image_size,
        num_encoder_layers=args.num_enc_layers,
        num_ope_iterative_steps=args.num_ope_iterative_steps,
        num_objects=args.num_objects,
        zero_shot=args.zero_shot,
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        kernel_dim=args.kernel_dim,
        backbone_name=args.backbone,
        swav_backbone=args.swav_backbone,
        train_backbone=args.backbone_lr > 0,
        reduction=args.reduction,
        dropout=args.dropout,
        layer_norm_eps=1e-5,
        mlp_factor=8,
        norm_first=args.pre_norm,
        activation=nn.GELU,
        norm=True,
        num_iterations =args.num_iterations 
    )
