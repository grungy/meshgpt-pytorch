from functools import partial
from math import pi
import torch.nn as nn
from meshgpt_pytorch.helpers import discretize

class EmbeddingMixin:
    def __init__(self, num_discrete_coors, dim_coor_embed, num_discrete_angle, dim_angle_embed, num_discrete_area, dim_area_embed, num_discrete_normals, dim_normal_embed, coor_continuous_range):
        self.num_discrete_coors = num_discrete_coors
        self.coor_continuous_range = coor_continuous_range

        self.discretize_face_coords = partial(discretize, num_discrete=num_discrete_coors, continuous_range=coor_continuous_range)
        self.coor_embed = nn.Embedding(num_discrete_coors, dim_coor_embed)

        self.discretize_angle = partial(discretize, num_discrete=num_discrete_angle, continuous_range=(0., pi))
        self.angle_embed = nn.Embedding(num_discrete_angle, dim_angle_embed)

        lo, hi = coor_continuous_range
        self.discretize_area = partial(discretize, num_discrete=num_discrete_area, continuous_range=(0., (hi - lo) ** 2))
        self.area_embed = nn.Embedding(num_discrete_area, dim_area_embed)

        self.discretize_normals = partial(discretize, num_discrete=num_discrete_normals, continuous_range=coor_continuous_range)
        self.normal_embed = nn.Embedding(num_discrete_normals, dim_normal_embed) 