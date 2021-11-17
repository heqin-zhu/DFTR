import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .SwinTransformer import SwinTransformer, Mlp, BasicLayer

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Sequential, nn.ModuleList)):
            weight_init(m)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.ReLU, nn.Softmax, nn.GELU, nn.Dropout, nn.LayerNorm)):
            pass
        else:
            pass


def embedding(x):
    B, C, H, W = x.size()
    x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
    return x


def unembedding(x, H, W):
    B, L, C = x.size()
    x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
    return x


class myTransBlk(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim=None,
            mlp_ratio=1,
            n_head=8,
            attn_pdrop=0,
            resid_pdrop=0,
            act_func=nn.GELU,
            depth=1,
              ):
        super().__init__()
        self.layers = BasicLayer(
                 dim=in_dim,
                 depth=depth,
                 num_heads=n_head,
                 window_size=7,
                 mlp_ratio=mlp_ratio,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                )
        if out_dim is not None and out_dim!=in_dim:
            self.final=nn.Sequential(
                                     nn.Linear(in_dim, out_dim),
                                     act_func(),
                                    )
        else:
            self.final = nn.Identity()
    def forward(self, x, H=None, W=None):
        if H is None or W is None:
            B, L, C = x.size()
            H = W = round(L**0.5)
        x_out, H, W, x, H, W = self.layers(x, H, W)
        return self.final(x_out)


class FAM(nn.Module):
    ''' Feature aggregation module

            x1: B x 4L x embed_dim
            x2: B x L x 2embed_dim
            out: B x 4L x embed_dim
    '''
    def __init__(self, embed_dim, mlp_ratio=4,n_head=8,attn_pdrop=0,resid_pdrop=0,depth=1,residual=False, act_func=nn.ReLU, multi_feature=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.residual = residual
        self.multi_feature = multi_feature

        self.in_trans = myTransBlk(
            in_dim=embed_dim, 
            out_dim=2*embed_dim, 
            mlp_ratio=mlp_ratio,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            act_func=act_func,
            depth=depth,
          )
        self.fuse_trans = myTransBlk(
            in_dim=2*(2+multi_feature)*embed_dim, 
            out_dim=embed_dim, 
            mlp_ratio=mlp_ratio,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            act_func=act_func,
            depth=depth,
          )

    def forward(self, x1, x2):
        ''' 
            x1: B x 4L x C
            x2: B x L x 2C
            out: B x 4L x C
        '''
        res = x1
        B, L1, C1 = x1.size()
        B, L2, C2 = x2.size()
        assert L1==4*L2 and 2*C1 == C2 == 2*self.embed_dim, f'dimension mismatch: {x1.size()}, {x2.size}'

        # x1: (B, 4L, C) -> (B, 4L, 2C)
        x1 = self.in_trans(x1)

        # x2: (B, L, 2C) -> (B, 4L, 2C)
        H = W = round(L2**0.5)
        x2 = unembedding(x2, H, W)
        x2 = F.interpolate(x2, size=(2*H, 2*W), mode='bilinear', align_corners=True)
        x2 = embedding(x2)

        x = torch.cat([x1,x2],dim=2).contiguous()
        if self.multi_feature:
            x = torch.cat((x, x1*x2),dim=2).contiguous()
        x = self.fuse_trans(x)
        if self.residual:
            x += res
        return x


class FFM(nn.Module):
    ''' Feature Fusion module

            x1: B x L x embed_dim
            x2: B x L x embed_dim
            out: B x L x embed_dim, B x L x embed_dim
    '''
    def __init__(self, embed_dim, mlp_ratio=4,n_head=8,attn_pdrop=0,resid_pdrop=0,depth=1,residual=False, act_func=nn.ReLU, multi_feature=True, num_fusion=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.residual = residual
        self.multi_feature = multi_feature

        self.shared_trans = myTransBlk(
            in_dim=(num_fusion+multi_feature)*embed_dim, 
            out_dim=embed_dim, 
            mlp_ratio=mlp_ratio,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
            act_func=act_func,
            depth=depth,
          )
        self.specific_trans = nn.ModuleList(
             [myTransBlk(
                in_dim=embed_dim, 
                out_dim=embed_dim, 
                mlp_ratio=mlp_ratio,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                act_func=act_func,
                depth=depth,
                )
            for i in range(num_fusion)
         ])

    def forward(self, *xs):
        ''' 
            x1: B x L x C
            x2: B x L x C
            out: B x L x C, B x L x C
        '''
        fuse = torch.cat(xs,dim=2).contiguous()
        if self.multi_feature:
            multi = 1
            for x in xs:
                multi *=x
            fuse = torch.cat((fuse, multi),dim=2).contiguous()
        fuse = self.shared_trans(fuse)
        ret = []
        for i,x in enumerate(xs):
            out = self.specific_trans[i](fuse)
            if self.residual:
                out+=x
            ret.append(out)
        return tuple(ret)

class DFTR(nn.Module):
    def __init__(self,
                 img_size=256,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 out_chans=1, 
                 residual=False,
                 multi_feature=True,
                 down_scale=4,
                 checkpoint_path='',
                 fusion_depth=1,
                 base_path='',
                 module_residual=False,
                ):
        super().__init__()
        self.base_path = base_path
        self.encoder = SwinTransformer(
                 img_size=img_size,
                 patch_size=patch_size,
                 in_chans=in_chans,
                 embed_dim=embed_dim,
                 depths=depths,
                 num_heads=num_heads,
                 window_size=window_size,
                 mlp_ratio=mlp_ratio,
                 qkv_bias=qkv_bias,
                 qk_scale=qk_scale,
                 drop_rate=drop_rate,
                 attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate,
                 norm_layer=nn.LayerNorm,
                 ape=ape,
                 patch_norm=patch_norm,
                 out_indices=out_indices,
                  )
        # reducing channel
        self.mask_trans = nn.ModuleList()
        self.depth_trans = nn.ModuleList()
        dim = embed_dim
        for i in range(len(depths)):
            self.mask_trans.append(nn.Sequential(nn.Linear(dim, dim//down_scale), nn.ReLU()))
            self.depth_trans.append(nn.Sequential(nn.Linear(dim, dim//down_scale), nn.ReLU()))
            dim*=2

        self.depth_fam = nn.ModuleList()
        self.mask_fam = nn.ModuleList()
        self.ffm = nn.ModuleList()

        dim = embed_dim//down_scale
        for i in range(len(depths)-1):
            self.depth_fam.append(FAM(dim, mlp_ratio=2,n_head=8,attn_pdrop=attn_drop_rate, resid_pdrop=drop_rate, depth=fusion_depth, residual=module_residual,act_func=nn.ReLU,multi_feature=multi_feature))
            self.mask_fam.append(FAM(dim, mlp_ratio=2,n_head=8,attn_pdrop=attn_drop_rate, resid_pdrop=drop_rate, depth=fusion_depth, residual=module_residual,act_func=nn.ReLU,multi_feature=multi_feature))
            self.ffm.append(FFM(dim, mlp_ratio=2,n_head=8,attn_pdrop=attn_drop_rate, resid_pdrop=drop_rate, depth=fusion_depth, residual=module_residual,act_func=nn.ReLU,multi_feature=multi_feature,num_fusion=2))
            dim *=2

        self.residual = residual

        self.mask_linear_lst =nn.ModuleList([nn.Linear(embed_dim//down_scale*(2**i), out_chans) for i in range(len(depths))])
        self.depth_linear_lst =nn.ModuleList([nn.Linear(embed_dim//down_scale*(2**i), out_chans) for i in range(len(depths))])
        
        self.checkpoint_path = checkpoint_path
        self.initialize()


    def forward(self, x, dest_size=None):
        B, C, H, W = x.shape
        features = self.encoder(x)
        mask_features = [down(embedding(f)) for f, down in zip(features, self.mask_trans)]
        depth_features = [down(embedding(f)) for f, down in zip(features, self.depth_trans)]

        mask_x = mask_features[-1]
        depth_x = depth_features[-1]

        mask_out = [mask_x]
        depth_out = [depth_x]

        for i in range(len(mask_features)-2,-1,-1):
            mask_x = self.mask_fam[i](mask_features[i], mask_x)
            depth_x = self.depth_fam[i](depth_features[i], depth_x)
            res_mask = mask_x
            res_depth = depth_x
            mask_x, depth_x = self.ffm[i](mask_x, depth_x)
            if self.residual:
                mask_x += res_mask
                depth_x += res_depth
            mask_out.append(mask_x)
            depth_out.append(depth_x)
        mask_out_unembed = []
        depth_out_unembed = []
        for i, (mask, depth)  in enumerate(zip(mask_out, depth_out)):
            m = unembedding(self.mask_linear_lst[len(mask_out)-i-1](mask), H*(2**i)//32, W*(2**i)//32)
            m = F.interpolate(m, size=(H,W), mode='bilinear', align_corners=True)
            mask_out_unembed.append(m)
            dep= unembedding(self.depth_linear_lst[len(depth_out)-i-1](depth), H*(2**i)//32, W*(2**i)//32)
            dep = F.interpolate(dep, size=(H,W), mode='bilinear', align_corners=True)
            depth_out_unembed.append(dep)
        if dest_size is not None:
            for li in [mask_out_unembed, depth_out_unembed]:
                for i in range(len(li)):
                    li[i]= F.interpolate(li[i], size=dest_size, mode='bilinear', align_corners=True)
        mask_out_unembed = mask_out_unembed[::-1]
        depth_out_unembed = depth_out_unembed[::-1]
        outputs = (*mask_out_unembed, *depth_out_unembed)
        return outputs
    def initialize(self):
        if os.path.exists(self.checkpoint_path):
            print(f'loading: {self.checkpoint_path}')
            self.load_state_dict(torch.load(self.checkpoint_path),strict=True)
        else:
            weight_init(self)
            if os.path.exists(self.base_path):
                print(f'loading: {self.base_path}')
                self.encoder.load_state_dict(torch.load(self.base_path)['model'],strict=False)
            else:
                print(f'Swin Transformer checkpoint doesn\'t exist: {self.base_path}')
