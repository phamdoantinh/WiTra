import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Callable

# ── Base Modules ────────────────────────────────────────────────────────────
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    if scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={self.drop_prob}, scale_by_keep={self.scale_by_keep}'
    
class layerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super(layerScale, self).__init__()
        self.gamma = nn.Parameter(init_values * torch.ones((dim)))

    def forward(self, x):
        return self.gamma * x
    
class Mlp(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = None,
            out_features: Optional[int] = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Optional[Callable] = None,
            bias: bool = True,
            drop: float = 0.
    ):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        bias = (bias, bias)
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            porj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Optional[Callable] = nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=porj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q * self.scale

        attn_logits = (q @ k.transpose(-2, -1))
        attn = attn_logits.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(
            self,
            dim : int,
            attn_cfg: dict,
            mlp_ratio: int = 4,
            act_layer: nn.Module = nn.GELU,
            norm_layer: Optional[Callable] = nn.LayerNorm,
            mlp_layer: Optional[Callable] = Mlp,
            drop_path: float = 0.,
            dropout: float = 0.,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, **attn_cfg)

        self.ls1 = layerScale(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop = dropout,  # Use regular dropout for the MLP projection
        )
        self.ls2 = layerScale(dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

    
class Embedding(nn.Module):
    def __init__(
            self,
            input_dim: int = 520,
            embed_dim: int = 768,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.proj = nn.Linear(1, embed_dim)
        
    def forward(self, x):
        # Input validation - replace inf/nan with safe values
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        
        B = x.shape[0]
        patches = x.view(B, self.input_dim, 1)
        emb = self.proj(patches)
        return emb
    


class EnCoder(nn.Module):
    def __init__(
            self,
            block_cfg: dict,
            input_dim: int = 520,
            embed_dim: int = 768,
            depth: int = 12,
            norm_layer: nn.Module = nn.LayerNorm,
            act_layer: nn.Module = nn.GELU,
            learn_pos: bool = True,
            dropout: float = 0.,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.depth = depth

        self.embedding = Embedding(input_dim=input_dim, embed_dim=embed_dim)
        
        self.cls_token_building = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_floor = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        if learn_pos:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_dim + 3, embed_dim))
        
        self.pos_drop = nn.Dropout(p=dropout)

        dpr = [x.item() for x in torch.linspace(0, dropout, depth)]
        
        self.blocks = nn.ModuleList([
            Block(
                dim      = embed_dim,
                attn_cfg = dict(block_cfg.get('attn_cfg', {})),  # copy — Block.pop() mutates
                mlp_ratio = block_cfg.get('mlp_ratio', 4),
                drop_path = dpr[i],
                dropout   = dropout,  # Propagate regular dropout to Transformer Blocks
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

    def _init_weights(self):
        if hasattr(self, 'pos_embed'):
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token_building, std=.02)
        nn.init.trunc_normal_(self.cls_token_floor, std=.02)
        nn.init.trunc_normal_(self.reg_token, std=.02)

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
        self.apply(init_weights)

    def forward(self, x):
        B = x.shape[0]
        emb =  self.embedding(x)

        cls_tokens_building = self.cls_token_building.expand(B, -1, -1)
        cls_tokens_floor = self.cls_token_floor.expand(B, -1, -1)
        reg_tokens = self.reg_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens_building, cls_tokens_floor, reg_tokens, emb), dim=1)

        if hasattr(self, 'pos_embed'):
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        cls_building = x[:, 0]
        cls_floor = x[:, 1]
        reg = x[:, 2]
        return cls_building, cls_floor, reg


class MultiHead(nn.Module):
    def __init__(
            self,
            dim: int,
            num_buildings: int,
            num_floors: int,
            dropout: float = 0.,
            num_bins: int = 100,
    ):
        super().__init__()
        self.dim = dim
        self.num_buildings = num_buildings
        self.num_floors = num_floors
        self.num_bins = num_bins

        # ── Floor & building heads (computed first — output used to condition coord) ──
        self.floor_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_floors)
        )
        self.building_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_buildings)
        )

        # ── Coord head: conditioned on floor soft probabilities ──────────────────
        # Input: reg_token (dim) ++ floor_softmax (num_floors)
        # Output: 2 * num_bins (logits for coordinate bin expectation)
        self.coord_head = nn.Sequential(
            nn.Linear(dim + num_floors, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 2 * num_bins)
        )
        
        # Pre-calculated bin centers in range [0, 1]
        self.register_buffer('bin_centers', torch.linspace(0.0, 1.0, num_bins))

    def _init_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(init_weights)

    def forward(self, cls_building, cls_floor, reg):
        # ── Step 1: classify floor & building ───────────────────────────────────
        floor    = self.floor_head(cls_floor)       # [B, num_floors]  — logits
        building = self.building_head(cls_building) # [B, num_buildings] — logits

        # softmax gives calibrated floor probabilities.
        # We intentionally DO NOT detach floor_soft so that the coordinate regression
        # loss backpropagates and acts as a spatial regularizer for floor classification.
        floor_soft   = F.softmax(floor, dim=-1)           # [B, num_floors]
        reg_cond     = torch.cat([reg, floor_soft], dim=-1)  # [B, dim + num_floors]

        coord_logits = self.coord_head(reg_cond)  # [B, 2 * num_bins]
        coord_logits = coord_logits.view(-1, 2, self.num_bins)  # [B, 2, num_bins]
        
        # Softmax over bins to get probability distribution
        coord_probs = F.softmax(coord_logits, dim=-1)  # [B, 2, num_bins]
        
        # Expectation: sum(probs * bin_centers)
        coord = torch.sum(coord_probs * self.bin_centers.to(coord_probs.dtype), dim=-1)  # [B, 2]

        return coord, floor, building

class SharedPrivateProjection(nn.Module):
    def __init__(self, dim: int, shared_dim: int = 128, private_dim: int = 128):
        super().__init__()
        self.dim = dim
        self.shared_dim = shared_dim
        self.private_dim = private_dim
        
        # Shared projection layer
        self.shared_proj = nn.Linear(dim, shared_dim)
        
        # Private projection layers
        self.private_proj_bld = nn.Linear(dim, private_dim)
        self.private_proj_flr = nn.Linear(dim, private_dim)
        self.private_proj_reg = nn.Linear(dim, private_dim)
        
        # Output projection layers (to map back to dim)
        self.out_proj_bld = nn.Linear(shared_dim + private_dim, dim)
        self.out_proj_flr = nn.Linear(shared_dim + private_dim, dim)
        self.out_proj_reg = nn.Linear(shared_dim + private_dim, dim)
        
    def _init_weights(self):
        # Initialize projections
        for m in [self.shared_proj, self.private_proj_bld, self.private_proj_flr, self.private_proj_reg,
                  self.out_proj_bld, self.out_proj_flr, self.out_proj_reg]:
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, h_bld, h_flr, h_reg):
        # 1. Project to shared space
        s_bld = self.shared_proj(h_bld)
        s_flr = self.shared_proj(h_flr)
        s_reg = self.shared_proj(h_reg)
        
        # 2. Project to private spaces
        p_bld = self.private_proj_bld(h_bld)
        p_flr = self.private_proj_flr(h_flr)
        p_reg = self.private_proj_reg(h_reg)
        
        # 3. Concatenate and project back to original dimension
        out_bld = self.out_proj_bld(torch.cat([s_bld, p_bld], dim=-1))
        out_flr = self.out_proj_flr(torch.cat([s_flr, p_flr], dim=-1))
        out_reg = self.out_proj_reg(torch.cat([s_reg, p_reg], dim=-1))
        
        # 4. Compute auxiliary losses
        def safe_cos_sim(x, y, eps=1e-8):
            # x, y shape: [B, D]
            # Use sqrt(sum(x^2) + eps) instead of norm(x).clamp(min=eps) to avoid NaN gradients at zero.
            x_norm = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
            y_norm = torch.sqrt(torch.sum(y**2, dim=-1) + eps)
            return torch.sum(x * y, dim=-1) / (x_norm * y_norm)
        
        # L_diff: Minimize similarity between different private subspaces (force orthogonality)
        # Using squared cosine similarity ensures gradients are smooth and penalizes both positive and negative correlation.
        l_diff = (safe_cos_sim(p_bld, p_flr)**2 + 
                  safe_cos_sim(p_bld, p_reg)**2 + 
                  safe_cos_sim(p_flr, p_reg)**2).mean()
        
        # L_sim: Shared similarity loss is set to 0.0.
        # The shared parameter projection layer (shared_proj) is sufficient to learn a shared representation,
        # and forcing continuous (reg) and categorical (bld, flr) token activations to align causes severe gradient conflict.
        l_sim = torch.tensor(0.0, device=h_bld.device)
        
        # L_sp_diff: Minimize similarity between shared and private subspace of the same task (disentanglement)
        l_sp_diff = (safe_cos_sim(s_bld, p_bld)**2 + 
                     safe_cos_sim(s_flr, p_flr)**2 + 
                     safe_cos_sim(s_reg, p_reg)**2).mean() / 3.0
        
        return out_bld, out_flr, out_reg, l_diff, l_sim, l_sp_diff

class Witra(nn.Module):
    def __init__(
            self,
            encoder_cfg: dict,
            multihead_cfg: dict,
    ):
        super().__init__()
        self.encoder = EnCoder(**encoder_cfg)
        self.multihead = MultiHead(**multihead_cfg)
        
        embed_dim = self.encoder.embed_dim
        self.shared_private_proj = SharedPrivateProjection(
            dim=embed_dim,
            shared_dim=embed_dim // 2,
            private_dim=embed_dim // 2
        )
        
        # Correctly initialize weights
        self.encoder._init_weights()
        self.shared_private_proj._init_weights()
        self.multihead._init_weights()
    
    def forward(self, x, return_ortho_loss: bool = False):
        cls_building, cls_floor, reg = self.encoder(x)
        cls_building, cls_floor, reg, l_diff, l_sim, l_sp_diff = self.shared_private_proj(
            cls_building, cls_floor, reg
        )
        coord, floor, building = self.multihead(cls_building, cls_floor, reg)
        if return_ortho_loss:
            return coord, floor, building, (l_diff, l_sim, l_sp_diff)
        else:
            return coord, floor, building
