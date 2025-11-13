from collections import defaultdict


import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils._pytree import tree_map

# logger = logging.get_logger(__name__)


def rotate_half(x):

    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def permute_qk(q, k):
    
    # NOTE: EasyLM and transformers use different method to compute rotary emebdding
    # we manually reorder the dim here to match our JAX implementation
    # which may not be optimal for speed
    # reference: https://github.com/young-geng/EasyLM/blob/981a2ed9630f44258a94b6f44dff2b7bd203ae8d/EasyLM/models/llama/convert_hf_to_easylm.py#L33
    bsz, num_head, seq_len, head_dim = q.shape
    q = q.reshape(bsz, num_head, seq_len, head_dim // 2, 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    k = k.reshape(bsz, num_head, seq_len, head_dim // 2, 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)

    return q, k


def undo_permute_qk(q, k):
    # NOTE: EasyLM and transformers use different method to compute rotary emebdding
    # we manually undo the reorder the dim here to match our JAX implementation
    # which may not be optimal for speed
    # reference: https://github.com/young-geng/EasyLM/blob/981a2ed9630f44258a94b6f44dff2b7bd203ae8d/EasyLM/models/llama/convert_hf_to_easylm.py#L33
    bsz, num_head, seq_len, head_dim = q.shape
    q = q.reshape(bsz, num_head, seq_len, 2, head_dim // 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)
    k = k.reshape(bsz, num_head, seq_len, 2, head_dim // 2).transpose(3, 4).reshape(bsz, num_head, seq_len, head_dim)

    return q, k


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=16,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float()/ self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):

        position_ids = position_ids.to(x.device)
  
        inv_freq = self.inv_freq.to(x.device)

        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "gpu" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)






def scan(f, init, xs, out, checkpoint_group=0):
    """Minic jax.lax.scan function."""
    carry = init
    if isinstance(xs, dict):
        num_items = len(next(iter(xs.values())))
    else:
        num_items = len(xs[0])

    def scan_fn(carry, i_start, i_end):
        for i in range(i_start, i_end):
            if isinstance(xs, dict):
                x = {key: tensor[i] for key, tensor in xs.items()}
            else:
                x = [x[i] for x in xs]
            carry, y = f(carry, x)
            out[i] = y
        return carry

    if checkpoint_group > 0:
        ckpt_every_n = num_items // checkpoint_group
        for k in range(0, num_items, ckpt_every_n):
            carry = torch.utils.checkpoint.checkpoint(
                scan_fn, carry, k, min(k + ckpt_every_n, num_items), use_reentrant=False
            )
    else:
        carry = scan_fn(carry, 0, num_items)

    return carry, out


#LayerNorm
def ln_fwd(x, gamma, beta, eps=1e-6):
    "Batch forward for LayerNorm."

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    return y



def ln_fused_l2_bwd(x, l2_target, gamma, beta, eps=1e-6):
    "Batch backward for LayerNorm fused with L2 loss."
    D = x.shape[-1]

    # Mean and variance computation
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalization
    std = torch.sqrt(var + eps)
    x_hat = (x - mu) / std

    # Scale and shift
    y = gamma * x_hat + beta

    grad_output = y - l2_target
    grad_x_hat = grad_output * gamma
    z = (
        (1.0 / D)
        * (
            D * grad_x_hat
            - grad_x_hat.sum(dim=-1, keepdim=True)
            - x_hat * (grad_x_hat * x_hat).sum(dim=-1, keepdim=True)
        )
        / std
    )

    return z



class DMAPCache:
    """
    DMAPCache is a data structure that holds the last hidden states and gradients for the DMAP layer.

    Arguments:
        model: DMAPModel
        batch_size: int

    Attributes:
        seqlen_offset: int
        mini_batch_size: int
        params_dict: Dict[str, Dict[int, torch.Tensor]]  *_states, *_grad -> # layer_idx -> [batch_size, ...]
        conv_states_dic: Dict[str, Dict[int, torch.Tensor]]  *_states -> # layer_idx -> [batch_size, ...]

    """

    def __init__(self, model, batch_size: int,mini_batch_size:int,device):
        # config = model.config
        self.seqlen_offset = 0
        self.mini_batch_size = mini_batch_size

        device = next(model.parameters()).device if device is None else device

        self.disable_update = False

        self.DMAP_params_dict = defaultdict(dict)
        self.DMAP_param_names = ["W1", "b1"]
        dict_weight={"W1":model.W1,"b1":model.b1}
      
        self.conv_states_dic = defaultdict(dict)
        for name in self.DMAP_param_names:
            weight=dict_weight[name]
            weight=torch.tile(weight.unsqueeze(0), (batch_size,) + (1,) * weight.dim()).to(device)
            self.DMAP_params_dict[f"{name}_states"] = weight.to(device)
            self.DMAP_params_dict[f"{name}_grad"] = torch.zeros_like(weight, device=device)

    def to(self, device):
        """将缓存中的所有张量移动到指定设备"""
        for name in self.DMAP_param_names:
            self.DMAP_params_dict[f"{name}_states"] = self.DMAP_params_dict[f"{name}_states"].to(device)
            self.DMAP_params_dict[f"{name}_grad"] = self.DMAP_params_dict[f"{name}_grad"].to(device)
        return self

    # 测试时参数更新
    def update(self, py_tree, seq_len):

        # # 不用更新, 直接return, 需要更新就注释掉
        # return

        device = next(iter(py_tree.values())).device
        if seq_len % self.mini_batch_size == 0:
            # copy last mini-batch states, clear gradients
            for name in self.DMAP_param_names:
                momentum = 0.9
                current_state = self.DMAP_params_dict[f"{name}_states"]
                new_state = py_tree[f"{name}_states"].detach()
                self.DMAP_params_dict[f"{name}_states"] = momentum * current_state + (1 - momentum) * new_state
                self.DMAP_params_dict[f"{name}_grad"] = torch.zeros_like(py_tree[f"{name}_states"], device=device)
        elif seq_len < self.mini_batch_size:
            if seq_len != 1 and self.seqlen_offset > 0 and self.seqlen_offset % self.mini_batch_size != 0:
                raise ValueError("fractional update not supported yet.")
            if (seq_len + self.seqlen_offset) % self.mini_batch_size == 0:
                # copy last mini-batch states, clear gradients
                for name in self.DMAP_param_names:
                    self.DMAP_params_dict[f"{name}_states"] = py_tree[f"{name}_states"].detach().to(device)
                    self.DMAP_params_dict[f"{name}_grad"] = torch.zeros_like(py_tree[f"{name}_states"], device=device)
            else:
                # copy gradients for the next update
                for name in self.DMAP_param_names:
                    self.DMAP_params_dict[f"{name}_grad"] = py_tree[f"{name}_grad"].detach().to(device)
        else:
            raise ValueError(f"seq_len {seq_len} is a partial update not supported yet")

    def DMAP_params_to_dict(self):
        return {name: self.DMAP_params_dict[name]for name in self.DMAP_params_dict}

# 推理态在线更新流程
class DMAPBase(nn.Module):
    def __init__(self, num_heads,hidden_size,mini_batch_size,rope_theta,device,stage):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size=hidden_size
        # self.v_dim=v_dim
        self.mini_batch_size=mini_batch_size
        self.rope_theta=rope_theta
        self.head_dim=hidden_size//num_heads
        self.DMAP_base_lr=1.0
        self.stage=stage
        assert self.stage in ['encoder','decoder']

        token_idx=1.0/torch.arange(1,self.mini_batch_size+1)
        self.register_buffer("token_idx",token_idx)
        self.learnable_token_idx=nn.Parameter(torch.zeros((self.mini_batch_size, )))
        self.post_norm=nn.LayerNorm(self.hidden_size,eps=1e-6).to(device)

        self._init_qkvo_proj()
        self._init_rope()
        self._init_DMAP_ln()
        self._init_DMAP_lr_gate()
        


    def _init_qkvo_proj(self):

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        # if self.stage==
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        

    def _init_rope(self):
        # self.rope_theta = self.config.rope_theta
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.mini_batch_size,
            base=self.rope_theta,
        )

    def _init_DMAP_lr_gate(self):
        linear_weight_data=nn.Linear(self.hidden_size,1,bias=True).weight.data
        linear_bias_data = nn.Linear(self.hidden_size,1,bias=True).bias.data

        self.learnable_DMAP_lr_weight=nn.Parameter(
            torch.stack(
                [torch.normal(0,0.02,size=linear_weight_data.shape) for _ in range(self.num_heads)],
                dim=0
            )
        )

        self.learnable_DMAP_lr_bias=nn.Parameter(
            torch.stack(
                [torch.zeros_like(linear_bias_data) for _ in range(self.num_heads)],
                dim=0
            )
        )

    def _init_DMAP_ln(self):
        ln_weight_data = nn.LayerNorm(self.head_dim).weight.data
        # prepending head dim -> [num_heads, width]
        self.DMAP_norm_weight = nn.Parameter(torch.tile(ln_weight_data.unsqueeze(0), (self.num_heads, 1)))
        ln_bias_data = nn.LayerNorm(self.head_dim).bias.data
        self.DMAP_norm_bias = nn.Parameter(torch.tile(ln_bias_data.unsqueeze(0), (self.num_heads, 1)))

    def get_qkv_projections(self, hidden_states,x2):
        # if x2!=None:
        #     print(x2.shape)
        # print(hidden_states.shape)
        XQ, XK = (
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            # self.v_proj(hidden_states),
        )
        if self.stage == 'decoder':
            assert x2 is not None
            XV = self.v_proj(x2)
        else:
            XV = self.v_proj(hidden_states)
        return XQ, XK, XV

    def get_eta(self, X, mini_batch_step_offset, mini_batch_size):
        # [B, num_heads, num_mini_batch, mini_batch_size, 1]
        DMAP_lr = torch.einsum("bnkc,hdc->bhnkd", X, self.learnable_DMAP_lr_weight) + self.learnable_DMAP_lr_bias.reshape(
            1, -1, 1, 1, 1
        )
        DMAP_lr = F.sigmoid(DMAP_lr)

        # [B, num_heads, num_mini_batch, 1, mini_batch_size]
        DMAP_lr = DMAP_lr.permute(0, 1, 2, 4, 3)
        DMAP_lr_eta = self.DMAP_base_lr * DMAP_lr / self.head_dim

        # [B, L]
        token_idx = self.token_idx + self.learnable_token_idx
        token_idx = token_idx[mini_batch_step_offset : mini_batch_step_offset + mini_batch_size]

        # token idx should be greast than 0
        token_idx = torch.clamp_min(token_idx, 0.0)

        # NOTE: token_eta is a scale factor that applies to each token in the mini-batch
        # [B, num_heads, num_mini_batch, mini_batch_size, 1]
        token_eta = torch.broadcast_to(
            token_idx.reshape(1, 1, 1, mini_batch_size, 1),
            (X.shape[0], self.num_heads, X.shape[1], mini_batch_size, 1),
        )

        return token_eta, DMAP_lr_eta

    # def apply_gate(self, hidden_states, DMAP_output):
    #     y = self.g_proj(hidden_states)
    #     # use 'tanh' approximation for matching JAX impl.
    #     y = F.gelu(y, approximate="tanh")
    #     output = y * DMAP_output
    #     return output

    def get_DMAP_inputs(self, inputs, mini_batch_size, cache_params):
        XQ = inputs["XQ"]
        XK = inputs["XK"]
        XV = inputs["XV"]
        X = inputs["X"]
        B, L, C = X.shape
        num_mini_batch = L // mini_batch_size
        # [B ,num_mini_batch, mini_batch_size, C]
        X = X.reshape(B, num_mini_batch, mini_batch_size, self.hidden_size)

        XQ = XQ.reshape(B, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)
        XK = XK.reshape(B, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)
        XV = XV.reshape(B, self.num_heads, num_mini_batch, mini_batch_size, self.head_dim)

        if cache_params is not None:
            mini_batch_step_offset = cache_params.seqlen_offset % self.mini_batch_size
        else:
            mini_batch_step_offset = 0 
        token_eta, DMAP_lr_eta = self.get_eta(X, mini_batch_step_offset, mini_batch_size)
        eta = token_eta * DMAP_lr_eta
        # decouple token_coeff and ilr_coeff for decoding
        inputs = {
            "XQ": XQ,
            "XK": XK,
            "XV": XV,
            "eta": eta,
            "token_eta": token_eta,
            "DMAP_lr_eta": DMAP_lr_eta,
        }
        return inputs

    def DMAP(
        self,
        inputs,
        mini_batch_size,
        last_mini_batch_params_dict,
        cache_params
    ):
        raise NotImplementedError("DMAP method must be implemented in DMAPBase subclasses.")

    
    def forward(
        self,
        hidden_states,
        x2,
        # attention_mask: Optional[torch.Tensor] = None,
        position_ids,
        cache_params
    ):
        # device = next(self.parameters()).device
        device = hidden_states.device
        hidden_states = hidden_states.to(device)
        position_ids = position_ids.to(device)
        self.to(device)
        B, L = hidden_states.shape[:2] # batch batch_size
        reminder_len = L % self.mini_batch_size
        num_mini_batch = L // self.mini_batch_size
        last_mini_batch_params_dict = None

        XQ, XK, XV = self.get_qkv_projections(hidden_states,x2)

        # [B, L, C] -> [B, L, num_heads, head_dim] -> [B, num_heads, L, head_dim]
        XQ = XQ.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XK = XK.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        XV = XV.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(XV, position_ids % self.mini_batch_size)

        # permute_qk and undo_permute_qk is just for aligning pytorch with jax pre-training
        XQ, XK = permute_qk(XQ, XK)
        XQ, XK = apply_rotary_pos_emb(XQ, XK, cos, sin)
        XQ, XK = undo_permute_qk(XQ, XK)

        output_hidden_states = []
        # when input sequence length is not a multiple of mini_batch_size
        # we need to compute them seperately, when computing the reminder,
        # we will need the last_mini_batch_params_dict to continue DMAP learning
        if num_mini_batch > 0:
            inputs = {
                "XQ": XQ[:, :, : num_mini_batch * self.mini_batch_size],
                "XK": XK[:, :, : num_mini_batch * self.mini_batch_size],
                "XV": XV[:, :, : num_mini_batch * self.mini_batch_size],
                "X": hidden_states[:, : num_mini_batch * self.mini_batch_size],
            }
            output_mod, last_mini_batch_params_dict = self.DMAP(
                self.get_DMAP_inputs(inputs, self.mini_batch_size, cache_params),
                mini_batch_size=self.mini_batch_size,
                last_mini_batch_params_dict=last_mini_batch_params_dict,
                cache_params=cache_params,
            )
            output_hidden_states.append(output_mod)
        if reminder_len > 0:
            inputs = {
                "XQ": XQ[:, :, -reminder_len:],
                "XK": XK[:, :, -reminder_len:],
                "XV": XV[:, :, -reminder_len:],
                "X": hidden_states[:, -reminder_len:],
            }
            output_reminder, _ = self.DMAP(
                self.get_DMAP_inputs(inputs, reminder_len, cache_params),
                mini_batch_size=reminder_len,
                last_mini_batch_params_dict=last_mini_batch_params_dict,
                cache_params=cache_params,
            )
            output_hidden_states.append(output_reminder)

        output_hidden_states = torch.cat(output_hidden_states, dim=1)
        output_hidden_states = self.post_norm(output_hidden_states)
        # if self.use_gate:
        #     output_hidden_states = self.apply_gate(hidden_states, output_hidden_states)
        output_hidden_states = self.o_proj(output_hidden_states)

        return output_hidden_states

# 测试时训练的在线更新参数
class DMAPLinear(DMAPBase):
    def __init__(self, num_heads,hidden_size,mini_batch_size,rope_theta,device,stage):
        super().__init__(num_heads,hidden_size,mini_batch_size,rope_theta,device,stage)
        # DMAP model initialization for DMAP-Linear
        # self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.num_heads, self.head_dim, self.head_dim)))
        self.W1 = nn.Parameter(torch.randn(self.num_heads, self.head_dim, self.head_dim)).to(device)
        self.b1 = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim)).to(device)
        self.device = device# self.post_norm=nn.LayerNorm(self.hidden_size,eps=1e-6).to(device)
        self.disable_update=True
    def check_device(self):
        """检查所有参数是否在同一设备上"""
        device = next(self.parameters()).device
        self.W1 = self.W1.to(device)
        self.b1 = self.b1.to(device)
        if hasattr(self, 'att_cache') and self.att_cache is not None:
            self.att_cache.to(device)
    def forward(self, hidden_states, x2, position_ids, cache_params):

        device = next(self.parameters()).device
        hidden_states = hidden_states.to(device)
        position_ids = position_ids.to(device)
        if x2 is not None:
            x2 = x2.to(device)
        if cache_params is not None:
            cache_params = cache_params.to(device)
        

        self.check_device()
        
        return super().forward(hidden_states, x2, position_ids, cache_params)
    def DMAP(
        self,
        inputs,
        mini_batch_size,
        last_mini_batch_params_dict,
        cache_params,
    ):
        # device = inputs["XV"].device 
        # inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
        #          for k, v in inputs.items()}
        if mini_batch_size is None:
            mini_batch_size = self.mini_batch_size

        # in this case, we are decoding
        if last_mini_batch_params_dict is None and cache_params is not None:
            last_mini_batch_params_dict = cache_params.DMAP_params_to_dict()

        # [B, num_heads, num_mini_batch, mini_batch_size, head_dim]
        B = inputs["XV"].shape[0]
        num_mini_batch = inputs["XV"].shape[2]
        L = inputs["XV"].shape[2] * inputs["XV"].shape[3]
        device = inputs["XV"].device
        dtype = inputs["XV"].dtype

        # NOTE:
        # for prefilling, we will always use dual form for faster computation
        # we need to use primal form if mini_batch_size is not a multiple of self.mini_batch_size
        # since we need store the gradient for the next mini-batch computation
        use_dual_form = cache_params is None or mini_batch_size % self.mini_batch_size == 0
        # use_dual_form = False
       


        def compute_mini_batch(params_dict, inputs):
            # [B, nh, f, f], nh=num_heads, f=head_dim
            W1_init = params_dict["W1_states"].detach()
            # [B, nh, 1, f]
            b1_init = params_dict["b1_states"].detach()

            # [B,nh,K,f], K=mini_batch_size
            XQ_mini_batch = inputs["XQ"].detach()
            XV_mini_batch = inputs["XV"].detach()
            XK_mini_batch = inputs["XK"].detach()
            # [B, nh, K, 1]
            eta_mini_batch = inputs["eta"].detach()
            token_eta_mini_batch = inputs["token_eta"].detach()
            DMAP_lr_eta_mini_batch = inputs["DMAP_lr_eta"].detach()

            X1 = XK_mini_batch


            # [B,nh,K,f] @ [B,nh,f,f] -> [B,nh,K,f]
            Z1 = X1 @ W1_init + b1_init
            reconstruction_target = XV_mini_batch - XK_mini_batch


            ln_weight = self.DMAP_norm_weight.reshape(self.num_heads, 1, self.head_dim)
            ln_bias = self.DMAP_norm_bias.reshape(self.num_heads, 1, self.head_dim)
            # [B,nh,K,f]
            with torch.no_grad():
                grad_l_wrt_Z1 = ln_fused_l2_bwd(Z1, reconstruction_target, ln_weight, ln_bias).detach()

 
            if use_dual_form:
                # [B,nh,K,K]
                Attn1 = torch.tril(XQ_mini_batch @ X1.transpose(-2, -1))
                # [B,nh,1,f] - [B,nh,K,K] @ [B,nh,K,f] -> [B,nh,K,f]
                b1_bar = b1_init - torch.tril(eta_mini_batch) @ grad_l_wrt_Z1
                # [B,nh,K,f] @ [B,nh,f,f] - ([B,nh,K,1] * [B,nh,K,K]) @ [B,nh,K,f] + [B,nh,K,f]
                Z1_bar = XQ_mini_batch @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar

                last_eta_mini_batch = eta_mini_batch[:, :, -1, :, None]
                # [B,nh,f,f] - [B,nh,f,K] @ [B,nh,K,f]
                W1_last = W1_init - (last_eta_mini_batch * X1).transpose(-1, -2) @ grad_l_wrt_Z1
                # [B,nh,1,f]
                b1_last = b1_init - torch.sum(last_eta_mini_batch * grad_l_wrt_Z1, dim=-2, keepdim=True)
                grad_W1_last = torch.zeros_like(W1_last)
                grad_b1_last = torch.zeros_like(b1_last)
            else:
                DMAP_lr_eta_mini_batch = torch.broadcast_to(
                    DMAP_lr_eta_mini_batch,
                    (
                        *DMAP_lr_eta_mini_batch.shape[:2],
                        mini_batch_size,
                        mini_batch_size,
                    ),
                )

                # [B, nh, K, f, f]
                grad_W1 = torch.einsum("bhki,bhkj->bhkij", X1, grad_l_wrt_Z1)
                grad_W1 = torch.einsum("bhnk,bhkij->bhnij", torch.tril(DMAP_lr_eta_mini_batch), grad_W1)
                grad_W1 = grad_W1 + params_dict["W1_grad"].unsqueeze(2).detach()
                # [B, nh, K, f]
                grad_b1 = torch.einsum("bhnk,bhki->bhni", torch.tril(DMAP_lr_eta_mini_batch), grad_l_wrt_Z1)
                grad_b1 = grad_b1 + params_dict["b1_grad"].detach()

                W1_bar = W1_init.unsqueeze(2) - grad_W1 * token_eta_mini_batch.unsqueeze(-1)
                b1_bar = b1_init - grad_b1 * token_eta_mini_batch

                # [B, nh, K, 1, f] @ [B, nh, K, f, f]
                Z1_bar = (XQ_mini_batch.unsqueeze(3) @ W1_bar).squeeze(3) + b1_bar

                W1_last = W1_bar[:, :, -1]
                b1_last = b1_bar[:, :, -1:]
                grad_W1_last = grad_W1[:, :, -1]
                grad_b1_last = grad_b1[:, :, -1:]

            Z1_bar = ln_fwd(Z1_bar, ln_weight, ln_bias)

            XQW_mini_batch = XQ_mini_batch + Z1_bar

            last_param_dict = {
                "W1_states": W1_last,
                "b1_states": b1_last,
                "W1_grad": grad_W1_last,
                "b1_grad": grad_b1_last,
            }
            return last_param_dict, XQW_mini_batch

        if last_mini_batch_params_dict is not None:
            init_params_dict = last_mini_batch_params_dict
        else:
            init_params_dict = {
                "W1_states": torch.tile(self.W1.unsqueeze(0), dims=(B, 1, 1, 1)),
                "b1_states": torch.tile(self.b1.unsqueeze(0), dims=(B, 1, 1, 1)),
            }
            init_params_dict.update(W1_grad=torch.zeros_like(init_params_dict["W1_states"]))
            init_params_dict.update(b1_grad=torch.zeros_like(init_params_dict["b1_states"]))

        # [B,num_heads, num_mini_batch, mini_batch_size, f] -> [num_mini_batch, B, num_heads, mini_batch_size, f]
        inputs = tree_map(lambda x: x.permute(2, 0, 1, 3, 4), inputs)

        # allocate output tensor
        XQW_batch = torch.empty(
            (num_mini_batch, B, self.num_heads, mini_batch_size, self.head_dim),
            device=device,
            dtype=dtype,
        )
        # XQW_batch: [num_mini_batch, B, num_heads, mini_batch_size, head_dim]
        batch_params_dict, XQW_batch = scan(
            compute_mini_batch,
            init_params_dict,
            inputs,
            XQW_batch,
            0
        )

        # [B, num_heads, L, C]
        if cache_params is not None:
            cache_params.update(batch_params_dict, L)

        # [num_mini_batch, B, num_heads, mini_batch_size, head_dim] -> [B, num_mini_batch, mini_batch_size, num_heads, head_dim]
        XQW_batch = XQW_batch.permute(1, 0, 3, 2, 4)
        # [B, L, C]
        XQW_batch = XQW_batch.reshape(B, L, self.hidden_size)
        return XQW_batch, batch_params_dict




