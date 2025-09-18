import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary
import time
import torch.nn.functional as F
from utils import *

import sys
global_dropout=0
class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)



class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) 
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.att_drop = nn.Dropout(global_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(global_dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (C // self.num_heads)**0.5
        attn = attn.softmax(dim=-1)
        attn = self.att_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(global_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VitModel(nn.Module):
    def __init__(self, in_channels, embed_dim, num_layers, num_heads, mlp_dim, num_classes, patch_size, in_size, output_dim=32, decoder=True,dropout_rate=0):
        super().__init__()

        # Logging
        self.config=get_config()
        global_dropout=dropout_rate
        # Model Configuration
        self.classifier_type=self.config["classifier_type"]   ## "patch_average"   "cls_token"
        self.in_channels = in_channels
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.input_resolution = in_size
        self.has_decoder = decoder
        self.num_of_patches = (in_size[0] // patch_size[0]) * (in_size[1] // patch_size[1])
        self.reduced_decay_factor = 1
        self.classification_scale = 100

        # Embedding Layers
        scale = embed_dim ** -0.5
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_of_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(global_dropout)

        # Transformer and Classification
        self.transformer = nn.Sequential(
            *[ViTBlock(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Parameter(scale * torch.randn(embed_dim, output_dim))
        self.classifier = nn.Linear(self.reduced_decay_factor * embed_dim, num_classes)

        # Decoder (if mask is True)
        if self.has_decoder:
            self.mask_token_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.ln_patch = LayerNorm(embed_dim)
            nn.init.normal_(self.mask_token_embedding, std=0.02)
            self.decoder = nn.Sequential(
                nn.Conv2d(in_channels=embed_dim, out_channels=patch_size[0] * patch_size[1], kernel_size=1)
            )

    def forward(self,x:torch.Tensor) :
        
        B, _, _, _ = x.shape
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
       
        x = x + self.pos_embed
      
        x = self.pos_drop(x)
        
        x = self.transformer(x)
       
        x_cls = self.norm(x[:, 0])
        x_patch=self.norm(x[:, 1:])
        #print(f'x_patch {x_patch.shape} x_cls {x_cls.shape}')
        #------------------------------------------------------------
        #cls_proj = F.normalize(x_cls @ self.proj,dim=-1)
        #x_patch_conv64=self.conv1d_64(x_patch)
        #x_cls_for_decoder=x_cls.unsqueeze(1)
        #print(f'x_patch_conv64=self.conv1d_64(x_patch) is {x_patch_conv64.shape}    cls dim {x_cls.shape}')
        #x_patch_for_decoder=self.conv1d_32_for_decoder(x_patch_conv64)
        #x_cls_for_decoder=self.conv1d_32_for_decoder(x_cls_for_decoder)
        #print(f'x_patch_conv64=self.conv1d_64(x_patch) is {x_patch_conv64.shape}    cls for decoder dim {x_cls_for_decoder.shape}')
        #print(f' x_patch_for_decoder=self.conv1d_32_for_decoder(x_patch_conv64) {x_patch_for_decoder.shape}')
        #x_patch_conv64=x_patch_conv64.squeeze(1)
        #print(f' x_patch_conv64=x_patch_conv64.squeeze(1) {x_patch_conv64.shape}')
        #print(f'-----------------------------------------------------------------------------------------------------')
        #----------------------------------------------------------------------------------------------
        #x_patch32 = x_patch.permute(0,2,1)
        #x_patch_conv32=self.conv1d_32(x_patch32)
       
         # Decoder path ---------------------------------------------------------
        if  self.has_decoder:
            decoder_in=x_patch
            B, L, C = decoder_in.shape
            H = W = int(L ** 0.5)
            x_patch_reshape = decoder_in.permute(0, 2, 1).reshape(B, C, 1, L)  
            x_rec = self.decoder(x_patch_reshape)
            x_rec=self.custom_pixel_shuffle(x_rec,self.patch_size[0],self.patch_size[1])
        else:
            x_rec=None
      
        #---------------------------------------------------------------------------------------
        #x_patch_conv32=x_patch_conv32.squeeze(1)
        #logits=self.classifier(x_cls)
        #logits=self.classifier(cls_proj)
        #logits=self.classifier(cls_proj)
        #logits=self.classifier(x_cls)
        #x_patch_conv64=x_patch_conv64.reshape(B,-1)
        #print(x_patch_conv64.shape)
        average_x_patch = torch.mean(x_patch, dim=1)
        average_patch_proj = F.normalize(average_x_patch @ self.proj,dim=-1)    
        #
        if self.classifier_type=="cls_token":
            logits=self.classifier(x_cls)
        elif self.classifier_type=="patch_average" :
            x_patch_flat=average_x_patch.reshape(B,-1)
            logits=self.classifier(x_patch_flat)    
        elif self.classifier_type=="proj_patch_average" :
              logits=self.classifier(average_patch_proj)       
        p1=0
        p2=0
        #print(f'x_rec {x_rec.shape}')
        
        #---------------------------------------------------------------------------------------------
        return {
        "logits": logits,
        "x_rec": x_rec,
        "x_patch": x_patch,
        "average_x_patch": average_x_patch,
        "average_patch_proj": average_patch_proj
       }
    def custom_pixel_shuffle(self,input, scale_h, scale_w):
        batch_size, channels, in_height, in_width = input.size()
        out_channels = channels // (scale_h * scale_w)
        if channels % (scale_h * scale_w) != 0:
            raise ValueError(f"Number of input channels ({channels}) must be divisible by scale_h * scale_w ({scale_h * scale_w}).")
        out_height = in_height * scale_h
        out_width = in_width * scale_w
        input = input.view(batch_size, out_channels, scale_h, scale_w, in_height, in_width)
        input = input.permute(0, 1, 4, 2, 5, 3).contiguous()
        output = input.view(batch_size, out_channels, out_height, out_width)
        return output 
    def pixel_shuffle_2d(self,input: torch.Tensor, scale_h: int, scale_w: int) -> torch.Tensor:
        """
        Pixel shuffle operation with separate height and width scaling factors.

        Args:
            input (torch.Tensor): Input tensor of shape (batch, channels, height, width)
            scale_h (int): Upscaling factor for height
            scale_w (int): Upscaling factor for width

        Returns:
            torch.Tensor: Output tensor with upscaled spatial dimensions
        """
        if input.dim() != 4:
            raise ValueError("Input tensor must be 4-dimensional (batch, channels, height, width).")

        batch_size, channels, in_height, in_width = input.shape
        out_channels = channels // (scale_h * scale_w)

        if channels % (scale_h * scale_w) != 0:
            raise ValueError(f"Number of input channels ({channels}) must be divisible by scale_h * scale_w ({scale_h * scale_w}).")

        out_height, out_width = in_height * scale_h, in_width * scale_w

        # Reshape to group channels accordingly
        input = input.reshape(batch_size, out_channels, scale_h, scale_w, in_height, in_width)

        # Permute to rearrange the pixels correctly
        input = input.permute(0, 1, 4, 2, 5, 3).reshape(batch_size, out_channels, out_height, out_width)

        return 
    

#--------------------------------------------------------------------------------------------------


def get_current_saved_model_vit(args,  decoder=True, checkpoints_path=None,config_model=None):
    device=get_device()
    dropout_rate = config_model.get("dropout", 0.3) if config_model else 0
    model = VitModel(
        in_channels=args.in_channels,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        num_classes=len(args.train_class_indices),
        in_size=args.in_size,
        output_dim=args.output_dim,
        decoder=decoder,
        dropout_rate=dropout_rate
    ).to(device)

    if checkpoints_path is not None:
        model = load_model(model, checkpoints_path, device)

    return model




