from __future__ import annotations

from pathlib import Path
from functools import partial
from math import ceil, pi, sqrt

import torch
from torch import nn, Tensor, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast

from pytorch_custom_utils import save_load

from beartype.typing import Tuple, Callable, List, Dict, Any
from meshgpt_pytorch.typing import Float, Int, Bool, typecheck

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from einx import get_at

from x_transformers import Decoder
from x_transformers.x_transformers import RMSNorm, FeedForward, LayerIntermediates

from x_transformers.autoregressive_wrapper import (
    eval_decorator,
    top_k,
    top_p,
)

from local_attention import LocalMHA

from vector_quantize_pytorch import (
    ResidualVQ,
    ResidualLFQ
)

from meshgpt_pytorch.data import derive_face_edges_from_faces
from meshgpt_pytorch.version import __version__

from taylor_series_linear_attention import TaylorSeriesLinearAttn

from classifier_free_guidance_pytorch import (
    classifier_free_guidance,
    TextEmbeddingReturner
)

from torch_geometric.nn.conv import SAGEConv

from gateloop_transformer import SimpleGateLoopLayer

from tqdm import tqdm

import logging

from meshgpt_pytorch.quantizer_mixin import QuantizerMixin

__all__ = ['exists', 'default', 'first', 'identity', 'divisible_by', 'is_odd', 'is_empty', 
           'is_tensor_empty', 'set_module_requires_grad_', 'l1norm', 'l2norm', 'safe_cat', 
           'pad_at_dim', 'pad_to_length', 'masked_mean', 'ContinuousEmbed', 'derive_angle',
           'get_derived_face_features', 'discretize', 'undiscretize', 'gaussian_blur_1d',
           'scatter_mean']

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(it):
    return it[0]

def identity(t, *args, **kwargs):
    return t

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def is_empty(x):
    return len(x) == 0

def is_tensor_empty(t: Tensor):
    return t.numel() == 0

def set_module_requires_grad_(
    module: Module,
    requires_grad: bool
):
    for param in module.parameters():
        param.requires_grad = requires_grad

def l1norm(t):
    return F.normalize(t, dim = -1, p = 1)

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

def safe_cat(tensors, dim):
    tensors = [*filter(exists, tensors)]

    if len(tensors) == 0:
        return None
    elif len(tensors) == 1:
        return first(tensors)

    return torch.cat(tensors, dim = dim)

def pad_at_dim(t, padding, dim = -1, value = 0):
    ndim = t.ndim
    right_dims = (ndim - dim - 1) if dim >= 0 else (-dim - 1)
    zeros = (0, 0) * right_dims
    return F.pad(t, (*zeros, *padding), value = value)

def pad_to_length(t, length, dim = -1, value = 0, right = True):
    curr_length = t.shape[dim]
    remainder = length - curr_length

    if remainder <= 0:
        return t

    padding = (0, remainder) if right else (remainder, 0)
    return pad_at_dim(t, padding, dim = dim, value = value)

def masked_mean(tensor, mask, dim = -1, eps = 1e-5):
    if not exists(mask):
        return tensor.mean(dim = dim)

    mask = rearrange(mask, '... -> ... 1')
    logger.debug("mask: ", mask, mask.shape)
    logger.debug("original tensor: ", tensor, tensor.shape)
    tensor = tensor.masked_fill(~mask, 0.)
    logger.debug("tensor: ", tensor, tensor.shape)

    total_el = mask.sum(dim = dim)
    num = tensor.sum(dim = dim)
    den = total_el.float().clamp(min = eps)
    mean = num / den
    mean = mean.masked_fill(total_el == 0, 0.)
    return mean

# continuous embed

def ContinuousEmbed(dim_cont):
    return nn.Sequential(
        Rearrange('... -> ... 1'),
        nn.Linear(1, dim_cont),
        nn.SiLU(),
        nn.Linear(dim_cont, dim_cont),
        nn.LayerNorm(dim_cont)
    )

# additional encoder features
# 1. angle (3), 2. area (1), 3. normals (3)

def derive_angle(x, y, eps = 1e-5):
    z = einsum('... d, ... d -> ...', l2norm(x), l2norm(y))
    return z.clip(-1 + eps, 1 - eps).arccos()

@torch.no_grad()
@typecheck
def get_derived_face_features(
    face_coords: Float['b nf nvf 3']  # 3 or 4 vertices with 3 coordinates
):
    is_quad = face_coords.shape[-2] == 4

    # shift face coordinates depending on triangles or quads
    # Shift face coords to create the set of edge vectors in one subtraction operation.
    shifted_face_coords = torch.roll(face_coords, 1, dims = (2,))

    angles  = derive_angle(face_coords, shifted_face_coords)

    if is_quad:
        # @sbriseid says quads need to be shifted by 2
        shifted_face_coords = torch.roll(shifted_face_coords, 1, dims = (2,))

    edge1, edge2, *_ = (face_coords - shifted_face_coords).unbind(dim = 2)

    cross_product = torch.cross(edge1, edge2, dim = -1)

    normals = l2norm(cross_product)
    area = cross_product.norm(dim = -1, keepdim = True) * 0.5

    return dict(
        angles = angles,
        area = area,
        normals = normals
    )   

# tensor helper functions

@typecheck
def discretize(
    t: Tensor,
    *,
    continuous_range: Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = (t - lo) / (hi - lo)
    t *= num_discrete
    t -= 0.5

    return t.round().long().clamp(min = 0, max = num_discrete - 1)

@typecheck
def undiscretize(
    t: Tensor,
    *,
    continuous_range = Tuple[float, float],
    num_discrete: int = 128
) -> Tensor:
    lo, hi = continuous_range
    assert hi > lo

    t = t.float()

    t += 0.5
    t /= num_discrete
    return t * (hi - lo) + lo

@typecheck
def gaussian_blur_1d(
    t: Tensor,
    *,
    sigma: float = 1.
) -> Tensor:

    _, _, channels, device, dtype = *t.shape, t.device, t.dtype

    width = int(ceil(sigma * 5))
    width += (width + 1) % 2
    half_width = width // 2

    distance = torch.arange(-half_width, half_width + 1, dtype = dtype, device = device)

    gaussian = torch.exp(-(distance ** 2) / (2 * sigma ** 2))
    gaussian = l1norm(gaussian)

    kernel = repeat(gaussian, 'n -> c 1 n', c = channels)

    t = rearrange(t, 'b n c -> b c n')
    out = F.conv1d(t, kernel, padding = half_width, groups = channels)
    return rearrange(out, 'b c n -> b n c')

@typecheck
def scatter_mean(
    tgt: Tensor,
    indices: Tensor,
    src = Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-5
):
    """
    todo: update to pytorch 2.1 and try https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html#torch.Tensor.scatter_reduce_
    """
    num = tgt.scatter_add(dim, indices, src)
    den = torch.zeros_like(tgt).scatter_add(dim, indices, torch.ones_like(src))
    return num / den.clamp(min = eps)