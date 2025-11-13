##############################################################################################
# The code is modified from ActionFormer: https://github.com/happyharrycn/actionformer_release
##############################################################################################
# online(DMAP):涉及DMAP都用这个backbone
import torch
from torch import nn
from torch.nn import functional as F

from .models import register_backbone,make_block
from .blocks import (get_sinusoid_encoding, TransformerBlock, MaskedConv1D,
                     ConvBlock, LayerNorm, GCNBlock,GCNAttetionBlock)
from .DMAP import DMAPLinear, DMAPCache

# action segmentation backbone with GCN
@register_backbone("convGCNTransformer")
class ConvGCNTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions and GCNs with transformers
    """
    def __init__(
        self,
        n_in,                  # input feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        gcn_type,              # gcn netwwork type
        use_gcn,               # whether to use GCN
        arch = (2, 2, 5),      # (#convs, #stem transformers, #branch transformers)
        mha_win_size = [-1]*6, # size of local window for mha
        scale_factor = 2,      # dowsampling rate for the branch
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
        use_rel_pe = False,    # use relative position embedding
        num_node = 20,
        
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(mha_win_size) == (1 + arch[2])
        self.n_in = n_in
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe
        self.use_gcn = use_gcn
        self.n_head = n_head
        # ---- DMAP online injection toggles ----
        self.DMAP_enabled = False
        # self.DMAP_inject_stage = 'stem'  # 'stem' | 'branch'
        self.DMAP_inject_stages = [] 
        self.DMAP_indices = []           # indices to apply; empty => all
        self.DMAP_mini_batch_size = 32
        self.DMAP_rope_theta = 10000
        self.DMAP_stage_kind = 'encoder' # 'encoder' or 'decoder'
        self.DMAP_module = None
        self.DMAP_cache = None
        # 新增：GCN 在线推理的窗口大小
        self.DMAP_gcn_window_size = -1   # -1 表示禁用在线模式

        # feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None
            graph_n_embd = n_embd // 8
            vis_n_embd = n_embd - graph_n_embd

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = vis_n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, vis_n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(vis_n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, vis_n_embd) / (vis_n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()

        # gcn
        stem_n_embd = vis_n_embd

        
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    stem_n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe
                )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    n_embd if use_gcn else stem_n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe
                )
            )
        if self.use_gcn:
            self.gcn = make_block(gcn_type,
                              **{'n_embd' : graph_n_embd,
                                'num_node' : num_node,})\
        
        # self.gcn = GCNBlock(graph_n_embd, num_node)
        # self.gcn = GCNAttetionBlock(graph_n_embd, num_node)
        
        # init weights
        self.apply(self.__init_weights__)
    
    def gcn_forward(self, bbox, bbox_class, edge_map, level):
        x_gcn = self.gcn(bbox, bbox_class, edge_map)
        fpn_gcn_feats = tuple()
        fpn_gcn_feats += (x_gcn, )
        for i in range(level):
            x_gcn = F.interpolate(x_gcn, scale_factor=0.5, mode='nearest')
            fpn_gcn_feats += (x_gcn, )
        return fpn_gcn_feats

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    

    def enable_DMAP(self, mini_batch_size=32, rope_theta=10000, inject_stages=None, indices=None, stage_kind='encoder', gcn_window_size=-1):
        self.DMAP_enabled = True
        self.DMAP_mini_batch_size = mini_batch_size
        self.DMAP_rope_theta = rope_theta
        self.DMAP_inject_stage = list(inject_stages) if inject_stages is not None else []
        self.DMAP_indices = list(indices) if indices is not None else []
        self.DMAP_stage_kind = stage_kind
        self.DMAP_gcn_window_size = gcn_window_size

    def _apply_DMAP(self, x, mask):
        # x: [B, C, T] -> DMAP expects [B, T, C]
        B, C, T = x.shape
        # print(f"[DEBUG] 1x-----: x.device={x.device}")
        device = x.device

        # 对齐已有 DMAP_module 到当前 device（设备变化时重置缓存）
        if self.DMAP_module is not None:
            try:
                module_device = next(self.DMAP_module.parameters()).device
            except StopIteration:
                module_device = device
            if module_device != device:
                self.DMAP_module.to(device)
                self.DMAP_cache = None  # 设备变化时缓存作废

        # lazy-init DMAP 模块（按通道数），并立即迁移到当前 device
        if (self.DMAP_module is None) or (getattr(self.DMAP_module, 'hidden_size', None) != C):
            self.DMAP_module = DMAPLinear(
                self.n_head, C, self.DMAP_mini_batch_size, self.DMAP_rope_theta, device, self.DMAP_stage_kind
            )
            self.DMAP_module.to(device)
            self.DMAP_cache = None  # 新模块创建后重置缓存

        # lazy-init/重建缓存（按 batch size 和设备）
        if (self.DMAP_cache is None) or (getattr(self.DMAP_cache, 'DMAP_params_dict', None) is None) or (self.DMAP_cache.DMAP_params_dict["W1_states"].shape[0] != B):
            self.DMAP_cache = DMAPCache(self.DMAP_module, batch_size=B, mini_batch_size=self.DMAP_mini_batch_size, device=device)

        position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        # print(f"[DEBUG] 3position_ids-----: position_ids.device={position_ids.device}")
        x_bt = x.transpose(1, 2)  # [B, T, C]
        # print(f"[DEBUG] 2x_bt-----: x_bt.device={x_bt.device}")

        out_bt = self.DMAP_module(x_bt, None, position_ids, self.DMAP_cache)  # [B, T, C]
        # print(f"[DEBUG] out_bt-----: out_bt.device={out_bt.device}")
        out_bt = out_bt.to(device)
        return out_bt.transpose(1, 2)  # [B, C, T]

    def forward(self, bbox, bbox_class, edge_map, x, mask):
        # bbox: batch, num_object, 4,time_len
        # bbox_class: batch,num_object, time_len
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # stem transformer
        for idx in range(len(self.stem)):
            if self.DMAP_enabled and (not self.training) and 'stem' in self.DMAP_inject_stages \
               and (len(self.DMAP_indices) == 0 or idx in self.DMAP_indices):
                # print(f"[DEBUG] before DMAP stem[{idx}]: x.device={x.device}")
                x = self._apply_DMAP(x, mask)    # 在线更新
                # print(f"[DEBUG] after  DMAP stem[{idx}]: x.device={x.device}")
            else:
                x, mask = self.stem[idx](x, mask)

        # gcn 拼接
        if self.use_gcn:

            # 新增：在线推理时，使用因果滑动窗口处理GCN
            if self.DMAP_enabled and (not self.training) and self.DMAP_gcn_window_size > 0:
                B, C_vis, T = x.shape # 获取 Batch, 视觉通道数, 时间长度
                # 获取GCN的输出通道数
                C_gcn = self.gcn.n_embd if hasattr(self.gcn, 'n_embd') else x.shape[1] // 7 # 从GCN模块获取或估算
                win_size = self.DMAP_gcn_window_size
                
                # 修复：创建一个形状正确的、用于存储GCN输出的张量
                # 形状应为 [B, C_gcn, T]
                x_gcn_full = torch.zeros((B, C_gcn, T), device=x.device, dtype=x.dtype)

                # 以滑动窗口的方式进行因果计算
                for t in range(T):
                    # 定义当前窗口的起始和结束点
                    start = max(0, t - win_size + 1)
                    end = t + 1
                    
                    # 对当前窗口的输入数据运行GCN
                    # 注意：GCN模块本身仍然一次性处理它接收到的切片
                    gcn_out_window = self.gcn(bbox[..., start:end], bbox_class[..., start:end], edge_map[..., start:end])
                    
                    # 只取窗口的最后一个时间点的输出，确保因果性
                    # gcn_out_window 的形状是 [B, C_gcn, window_length]
                    # 我们需要的是最后一个时间步，即索引为 -1
                    x_gcn_full[..., t] = gcn_out_window[..., -1]
                
                x_gcn = x_gcn_full
            else:
                x_gcn = self.gcn(bbox, bbox_class, edge_map)
            # print(f"[DEBUG] x.device = {x.device}")
            # print(f"[DEBUG] x_gcn.device = {x_gcn.device}")
            x = torch.cat((x, x_gcn), dim=1)

        out_feats = (x, ); out_masks = (mask, )

        # branch
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            if self.DMAP_enabled and (not self.training) and 'branch' in self.DMAP_inject_stages \
               and (len(self.DMAP_indices) == 0 or idx in self.DMAP_indices):
                
                x = self._apply_DMAP(x, mask)    # 在线更新
            out_feats += (x, ); out_masks += (mask, )
        return out_feats, out_masks

@register_backbone("convTransformer")
class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        n_in,                  # input feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        arch = (2, 2, 5),      # (#convs, #stem transformers, #branch transformers)
        mha_win_size = [-1]*6, # size of local window for mha
        scale_factor = 2,      # dowsampling rate for the branch
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
        use_rel_pe = False,    # use relative position embedding
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(mha_win_size) == (1 + arch[2])
        self.n_in = n_in
        self.arch = arch
        self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe

        # feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe
                )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe
                )
            )
        
        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, bbox, bbox_class, edge_map, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # stem transformer
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)
        
        # prep for outputs
        out_feats = (x, )
        out_masks = (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )


        return out_feats, out_masks
