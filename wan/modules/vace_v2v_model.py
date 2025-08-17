import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.configuration_utils import register_to_config

from .model import WanAttentionBlock, WanModel, sinusoidal_embedding_1d

class CustomVaceWanModel(WanModel):
    def __init__(self,
                 vace_layers=None,
                 vace_in_dim=None,
                 model_type='vace',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        super().__init__(model_type, patch_size, text_len, in_dim, dim, ffn_dim,
                         freq_dim, text_dim, out_dim, num_heads, num_layers,
                         window_size, qk_norm, cross_attn_norm, eps)
        self.vace_layers = vace_layers if vace_layers is not None else []
        self.vace_in_dim = self.in_dim if vace_in_dim is None else vace_in_dim
        
        # Create vace_layers_mapping if vace_layers is provided
        self.vace_layers_mapping = {}
        if self.vace_layers:
            for idx, layer_num in enumerate(self.vace_layers):
                self.vace_layers_mapping[layer_num] = idx
        
        # Vace WanModel blocks
        self.blocks = nn.ModuleList([
            BaseWanAttentionBlock(
                't2v_cross_attn',
                self.dim,
                self.ffn_dim,
                self.num_heads,
                self.window_size,
                self.qk_norm,
                self.cross_attn_norm,
                self.eps,
                block_id=self.vace_layers_mapping.get(i, None))
            for i in range(self.num_layers)
        ])


        # vace blocks
        self.vace_blocks = nn.ModuleList([
            VaceWanAttentionBlock(
                't2v_cross_attn',
                self.dim,
                self.ffn_dim,
                self.num_heads,
                self.window_size,
                self.qk_norm,
                self.cross_attn_norm,
                self.eps,
                block_id=i) for i in self.vace_layers
        ]) if self.vace_layers else nn.ModuleList()

        # vace patch embedding (adapted for Wan2.1 checkpoint)
        # Checkpoint has shape [1536, 96, 1, 2, 2], so in=96, out=1536
        self.vace_patch_embedding = nn.Conv3d(
            96,  # in_channels from checkpoint
            1536,  # out_channels from checkpoint
            kernel_size=self.patch_size,
            stride=self.patch_size)
        # Projection layer to match vace_blocks dimension (self.dim)
        self.vace_proj = nn.Linear(1536, self.dim)
        
    def forward(
        self,
        x,
        t,
        vace_context,
        context,
        seq_len,
        vace_context_scale=1.0,
    ):
        # =================================================================
        # HACK: Adapt 16-channel input from Wan2.1 VAE to 48-channel Wan2.2 Model
        # 1. Pad input from 16 to 48 channels
        x_padded = []
        for u in x:
            # Correctly create a padding tensor of shape (32, D, H, W)
            pad_shape = (32, *u.shape[1:])
            pad = torch.zeros(pad_shape, device=u.device, dtype=u.dtype)
            # Correctly concatenate along the channel dimension (dim=0)
            padded_u = torch.cat([u, pad], dim=0)
            x_padded.append(padded_u)
        x = x_padded
        # =================================================================

        # 디바이스 정렬 : 패치 임베딩 모델 - 주파수값 동일 위치에 정렬.
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # 패치 임베딩 : 동영상 -> 패치 -> 텐서
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # freq 임베딩 : self.freqs (주파수값) -> 주파수값
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32
        
        # context 2차 임베딩
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # args
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)
        kwargs['context_scale'] = vace_context_scale
        
        # VACE 과정
        hints = self.forward_vace(x, vace_context, seq_len, kwargs)
        kwargs['hints'] = hints

        # VACEWanModel 통과
        for block in self.blocks:
            x = block(x, **kwargs)

        # 텐서 -> 패치 -> 동영상
        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        
        # =================================================================
        # HACK: Adapt 48-channel output from Wan2.2 Model to 16-channel for Wan2.1 VAE
        # 2. Truncate output from 48 to 16 channels
        x_truncated = [u[:16, ...] for u in x]
        x = x_truncated
        # =================================================================

        return [u.float() for u in x]

    def forward_vace(self, x, vace_context, seq_len, kwargs):
        # HACK: Pad vace_context to 96 channels to match the adapted layer
        vace_context_padded = []
        for u in vace_context:
            # Assuming input is 80 channels (32 from inactive/reactive + 48 from mask)
            # This might need adjustment if the assumption is wrong.
            num_channels = u.shape[0]
            if num_channels < 96:
                pad_channels = 96 - num_channels
                pad_shape = (pad_channels, *u.shape[1:])
                pad = torch.zeros(pad_shape, device=u.device, dtype=u.dtype)
                padded_u = torch.cat([u, pad], dim=0)
                vace_context_padded.append(padded_u)
            else:
                vace_context_padded.append(u)
        c = vace_context_padded

        # embeddings
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in c]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in c
        ])
        # Apply projection to match dimensions
        c = self.vace_proj(c)

        # arguments
        new_kwargs = dict(x=x)
        new_kwargs.update(kwargs)

        hints = []
        for block in self.vace_blocks:
            c, c_skip = block(c, **new_kwargs)
            hints.append(c_skip)
        return hints

class VaceWanAttentionBlock(WanAttentionBlock):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 block_id=0):
        super().__init__(dim, ffn_dim, num_heads, window_size,
                         qk_norm, cross_attn_norm, eps)
        self.cross_attn_type = cross_attn_type
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(self.dim, self.dim)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(self.dim, self.dim)
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, c, x, **kwargs):
        if self.block_id == 0:
            c = self.before_proj(c) + x

        c = super().forward(c, **kwargs)
        c_skip = self.after_proj(c)
        return c, c_skip

class BaseWanAttentionBlock(WanAttentionBlock):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 block_id=None):
        super().__init__(dim, ffn_dim, num_heads, window_size,
                         qk_norm, cross_attn_norm, eps)
        self.cross_attn_type = cross_attn_type
        self.block_id = block_id

    def forward(self, x, hints, context_scale=1.0, **kwargs):
        x = super().forward(x, **kwargs)
        if self.block_id is not None:
            x = x + hints[self.block_id] * context_scale
        return x
