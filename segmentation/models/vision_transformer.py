import torch
import torch.nn as nn

from .base_module import BaseModule
from ..builder import MODELS

class DropPath(BaseModule):
    def __init__(self , drop_ratio=0.):
        super().__init__()
        self.drop_ratio = drop_ratio
        
    def forward(self, x):
        input_ = x
        if self.drop_ratio == 0.:
            return input_
        keep_prob = 1 - self.drop_ratio
        shape = (x.shape[0] , ) + (1,) * (input_.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape , dtype=input_.dtype , devic=input_.dtype)
        random_tensor.floor_()      
        result_ = input_.div(keep_prob)
        return result_

class PatchEmbedding(BaseModule):
    '''
        Convert image to patch embedding
    '''
    def __init__(self , image_size=224 , patch_size=16 , in_channels=3 , embed_dim=768 ,norm_layer=None) -> None:
        super().__init__()
        self.image_size = (image_size , image_size)
        self.patch_size = (patch_size , patch_size)
        self.num_patches = (self.image_size[0] / self.patch_size[0]) * (self.image_size[1] / self.patch_size[1])
        self.embedConv2d = nn.Conv2d(in_channels , embed_dim , patch_size , patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self , x):
        input_ = x
        b , c , h , w = input_.shape
        assert (h == self.image_size[0] and w == self.image_size[1]) , \
            f"Input image size ({h}*{w}) doesn't match model embedding size({self.image_size[0]}*{self.image_size[1]})."
        temp_ = self.embedConv2d(input_)
        result_ = self.norm(temp_)
        return result_
        
class MLPBlock(BaseModule):
    def __init__(self , in_channels , hidden_channels=None , out_channels=None , drop_ratio=0.):
        super().__init__()
        self.baseConvBlock = nn.Sequential(
            # B * 197 * 768 => B * 197 * 3072
            nn.Linear(in_channels , hidden_channels),
            nn.GELU(),
            nn.Dropout(drop_ratio),
            # B * 197 * 3072 => B * 197 * 768
            nn.Linear(hidden_channels , out_channels),
            nn.Dropout(drop_ratio),
        )
    
    def forward(self, x):
        return self.baseConvBlock(x)

class AttentionBlock(BaseModule):
    def __init__(self , dim , num_heads , qkv_bias=False , qkv_scale=None , 
                 attention_drop_ratio=0. , final_drop_ratio=0.) -> None:
        """
            num_heads (int): number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.qkv = nn.Linear(dim , dim * 3 ,bias=qkv_bias)
        self.scale = qkv_scale or (self.head_dim ** -0.5)
        self.attention_drop = nn.Dropout(attention_drop_ratio)
        self.final_linear = nn.Linear(dim , dim)
        self.final_drop = nn.Dropout(final_drop_ratio)
         
    def forward(self, x):
        # B * (num_patches + 1) * total_embed_dim
        input_ = x 
        b , n , c = input_.shape
        # qkv : B * (num_patches + 1) * total_embed_dim => B * (num_patches + 1) * (3 * total_embed_dim)
        # reshape : => B * (num_patches + 1) * 3 * num_heads * embed_dim_per_head
        # permute : => 3 * B * num_heads * (num_patches + 1) * embed_dim_per_head
        matrix_list = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q , k , v = matrix_list[0] , matrix_list[1] , matrix_list[2]
        
        # transpose : B * num_heads * embed_dim_per_head * (num_patches + 1) 
        # torch.mul : => B * num_heads *  (num_patches + 1) * (num_patches + 1) 
        temp = torch.mul(q , k.transpose(-2,-1)) * self.scale
        temp = temp.softmax(dim = -1)
        temp = self.attention_drop(temp)
        
        # reshape : => B * (num_patches + 1) * total_embed_dim
        temp = torch.mul(temp , v).transpose(1 , 2).reshape(b , n , c)
        temp = self.final_linear(temp)
        result_ = self.final_drop(temp)
        return result_
    
class EncoderBlock(BaseModule):
    def __init__(self , dim , num_heads , mlp_ratio=4. , qkv_bias=False , qk_scale=None , 
                 drop_ratio=0. , attention_drop_ratio=0. , drop_path_ratio=0.):
        super(EncoderBlock , self).__init__()
        self.attention_branch = nn.Sequential(
            nn.LayerNorm(), 
            AttentionBlock(dim , num_heads , qkv_bias , qk_scale , attention_drop_ratio , drop_ratio), 
            DropPath(drop_path_ratio)
        )
        
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        
        self.mlp_branch = nn.Sequential(
            nn.LayerNorm(),
            MLPBlock(dim , hidden_channels=self.mlp_hidden_dim , drop_ratio=drop_ratio),
            DropPath(drop_path_ratio)
        )
        
    def forward(self, x):
        input_ = x
        temp = self.attention_branch(input_) + input_
        result_ = self.mlp_branch(temp) + temp
        return result_

    
@MODELS.register_module()
class VisionTransformer(BaseModule):
    def __init__(self , image_size=224 , patch_size=16 , num_classes=1000 ,
                 embed_dim=768 , encoder_layers=12 , num_heads=12 , mlp_ratio=4 ,
                 qkv_bias=True , qk_scale=None , distilled=False , drop_ratio=0 , 
                 attention_drop_ratio=0. , drop_path_ratio=0. , representation_size = None ,
                 embed_layer=None , act_func=None) -> None:
        """
            num_heads (int): number of attention heads
        """
        super(VisionTransformer , self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.embed_layer = embed_layer if embed_layer else PatchEmbedding
        self.act_func = act_func if act_func else nn.GELU       
        
        self.patch_embedding = self.embed_layer(image_size , patch_size , 3 , embed_dim)
        self.num_patches = self.patch_embedding.num_patches # P^2
        
        self.num_tokens = 2 if distilled else 1
        self.cls_token = nn.Parameter(torch.zeros(1 , 1 , embed_dim))
        nn.init.trunc_normal_(self.cls_token , std=0.02)
        self.dist_token = nn.Parameter(torch.zeros(1 , 1 , embed_dim)) if distilled else None
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token , std=0.02)
        
        # Position Embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1 , self.num_patches + self.num_tokens , embed_dim))
        nn.init.trunc_normal_(self.pos_embedding , std=0.02)
        self.pos_drop = nn.Dropout(drop_ratio)
        
        # stochastic depth decay rule
        random_drop_ratio = [x.item() for x in torch.linspace(0 , drop_path_ratio , encoder_layers)]
        # Transformer Encoder + Layer Norm
        self.skeleton_ = nn.Sequential(*[
            EncoderBlock(dim=embed_dim , num_heads=num_heads , mlp_ratio=mlp_ratio , qkv_bias=qkv_bias , 
                        qk_scale=qk_scale , drop_ratio=drop_ratio , attention_drop_ratio=attention_drop_ratio , 
                        drop_path_ratio=random_drop_ratio[i] ,) 
            for i in range(encoder_layers) ] ,
           nn.LayerNorm(embed_dim) 
        )
        
        # Pre-Logits
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim , representation_size),
                nn.Tanh()
            )
        else:
            self.pre_logits = nn.Identity()
        
        # Classifier heads
        self.cls_head = nn.Linear(self.num_features , self.num_classes) if num_classes > 0 else nn.Identity()
        if distilled:
            self.dist_head = nn.Linear(self.embed_dim , self.num_classes) if num_classes > 0 else nn.Identity()
        
        
    def forward(self, x):
        input_ = x
        # B * 224 * 224 * 3 => B * 196 * 768 
        patch_embed = self.patch_embedding(input_)
        # 1 * 1 * 768 => B * 1 * 768
        cls_token = self.cls_token.expand(input_.shape[0] , -1 , -1)
        
        temp = patch_embed
        if self.dist_token is not None:
            dist_token = self.dist_token.expand(input_.shape[0] , -1 , -1)
            temp = torch.cat((cls_token , dist_token , temp) , 1)
        else:
            torch.cat((cls_token , temp) , 1)
        
        temp = self.pos_drop(temp + self.pos_embedding)
        
        # Enter encoder
        temp = self.skeleton_(temp)
        
        # Enter Pre-Logits Classifier heads
        if self.dist_token is not None:
            temp_ = temp[: , 0] , temp[: , 1]
            result , dist_result = self.cls_head(temp_[0]) , self.dist_head(temp_[1])
            result_ = (result + dist_result) / 2
        else:
            temp = self.pre_logits(temp[:,0])
            result_ = self.cls_head(temp)
            
        
        return result_

@MODELS.register_module()
class VisionTransformer_base_patch16_r224(BaseModule):
    def __init__(self , num_classes=1000):
        super().__init__()
        self.model = VisionTransformer(
            image_size=224 , patch_size=16 , embed_dim=768 , 
            encoder_layers=12 , num_heads=12 , representation_size=None , num_classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)
    
@MODELS.register_module()
class VisionTransformer_base_patch16_r224_in21k(BaseModule):
    def __init__(self , num_classes=21843 , has_logit=True):
        super().__init__()
        self.model = VisionTransformer(
            image_size=224 , patch_size=16 , embed_dim=768 , 
            encoder_layers=12 , num_heads=12 , 
            representation_size=768 if has_logit else None , 
            num_classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)

@MODELS.register_module()
class VisionTransformer_base_patch32_r224(BaseModule):
    def __init__(self , num_classes=1000):
        super().__init__()
        self.model = VisionTransformer(
            image_size=224 , patch_size=32 , embed_dim=768 , 
            encoder_layers=12 , num_heads=12 , representation_size=None , num_classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)

@MODELS.register_module()
class VisionTransformer_base_patch32_r224_in21k(BaseModule):
    def __init__(self , num_classes=21843 , has_logit=True):
        super().__init__()
        self.model = VisionTransformer(
            image_size=224 , patch_size=32 , embed_dim=768 , 
            encoder_layers=12 , num_heads=12 , 
            representation_size=768 if has_logit else None , 
            num_classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)

@MODELS.register_module()
class VisionTransformer_large_patch16_r224(BaseModule):
    def __init__(self , num_classes=1000):
        super().__init__()
        self.model = VisionTransformer(
            image_size=224 , patch_size=16 , embed_dim=1024 , 
            encoder_layers=24 , num_heads=16 , representation_size=None , num_classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)

@MODELS.register_module()
class VisionTransformer_large_patch16_r224_in21k(BaseModule):
    def __init__(self , num_classes=21843 , has_logit=True):
        super().__init__()
        self.model = VisionTransformer(
            image_size=224 , patch_size=16 , embed_dim=1024 , 
            encoder_layers=24 , num_heads=16 , 
            representation_size=1024 if has_logit else None , 
            num_classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)

@MODELS.register_module()
class VisionTransformer_large_patch32_r224_in21k(BaseModule):
    def __init__(self , num_classes=21843 , has_logit=True):
        super().__init__()
        self.model = VisionTransformer(
            image_size=224 , patch_size=32 , embed_dim=1024 , 
            encoder_layers=24 , num_heads=16 , 
            representation_size=1024 if has_logit else None , 
            num_classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)

@MODELS.register_module()
class VisionTransformer_huge_patch14_r224_in21k(BaseModule):
    def __init__(self , num_classes=21843 , has_logit=True):
        super().__init__()
        self.model = VisionTransformer(
            image_size=224 , patch_size=14 , embed_dim=1280 , 
            encoder_layers=32 , num_heads=16 , 
            representation_size=1280 if has_logit else None , 
            num_classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = VisionTransformer()
    