import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from relational_dynamics.model.pointconv_util_groupnorm import PointConvDensitySetAbstraction

class EmbeddingNetTorch(nn.Module):
    def __init__(
            self, n_objects: int,
            width: int, layers: int, heads: int, input_feature_num: int,
            d_hidden: int, n_unary: int, n_binary: int,
            dim_feedforward: int, 
            use_discrete_place = False,
            latent_discrete_continuous = False, 
            seperate_discrete_continuous = False,
            simple_encoding = False,
            transformer_dynamics = False,
            torch_embedding = False,
            complicated_pre_dynamics = False,
            train_env_identity = False, 
            total_env_identity = 1,
            one_bit_env = False,
            direct_transformer = False,
            enable_high_push = False,
            enable_place_inside = False,
            seperate_place = False,
            use_seperate_latent_embedding = True,
            seperate_action_emb = False,
            use_mlp_encoder = False, 
        ): 
        super(EmbeddingNetTorch, self).__init__()
        d_input = width

        self.use_seperate_latent_embedding = use_seperate_latent_embedding

        self.train_env_identity = train_env_identity

        self.direct_transformer = direct_transformer

        self.enable_high_push = enable_high_push

        self.enable_place_inside = enable_place_inside

        self.seperate_place = seperate_place

        self.seperate_action_emb = seperate_action_emb

        self.seperate_discrete_continuous = seperate_discrete_continuous

        self.simple_encoding = simple_encoding

        self.transformer_dynamics = transformer_dynamics

        self.torch_embedding = torch_embedding

        self.complicated_pre_dynamics = complicated_pre_dynamics
        
        encoder_layers = TransformerEncoderLayer(width, heads, batch_first = True, dim_feedforward = dim_feedforward)

        self.transformer = TransformerEncoder(encoder_layers, layers)

        self.one_hot_encoding_dim = (int)(width/2) 

        self.one_hot_encoding_embed = nn.Sequential(
            nn.Embedding(n_objects, self.one_hot_encoding_dim)
        )

        self.continuous_action_emb = nn.Sequential(
            nn.Linear(2, self.one_hot_encoding_dim),
            nn.ReLU(),
            nn.Linear(self.one_hot_encoding_dim, self.one_hot_encoding_dim)
        )
        self.continuous_action_emb_1 = nn.Sequential(
            nn.Linear(2, self.one_hot_encoding_dim),
            nn.ReLU(),
            nn.Linear(self.one_hot_encoding_dim, self.one_hot_encoding_dim)
        )
      
        encoder_layers_1 = TransformerEncoderLayer(width, heads, batch_first = True, dim_feedforward = dim_feedforward)
        self.graph_dynamics_0 = TransformerEncoder(encoder_layers_1, layers)
        encoder_layers_2 = TransformerEncoderLayer(width, heads, batch_first = True, dim_feedforward = dim_feedforward)
        self.graph_dynamics_1 = TransformerEncoder(encoder_layers_2, layers)

        self.n_unary = n_unary
        self.n_binary = n_binary
        self.d_hidden = d_hidden

        self.f_unary = self.get_head_unary(d_input, d_hidden, 1, n_unary)
        self.f_binary = self.get_head(d_input, d_hidden, 2, n_binary)
        

        self.ln_post = nn.LayerNorm(width)

    def get_head(self, d_input, d_hidden, n_args, n_binary):
        if d_hidden > 1:
            head = nn.Sequential(
                nn.Linear(d_input * n_args, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, n_binary),
                nn.Sigmoid()
            )
        else:
            head = nn.Sequential(
                nn.Linear(d_input * n_args, d_hidden),
                nn.Sigmoid()
            )
        return head

    def get_head_unary(self, d_input, d_hidden, n_args, n_unary):
        if d_hidden > 1:
            head = nn.Sequential(
                nn.Linear(d_input * n_args, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, n_unary)
            )
        else:
            head = nn.Sequential(
                nn.Linear(d_input * n_args, d_hidden)
            )
        return head

class QuickReadoutNet(nn.Module):
    def __init__(
            self, n_objects: int,
            width: int, layers: int, heads: int, input_feature_num: int,
            d_hidden: int, n_unary: int, n_binary: int,
            dim_feedforward: int, 
            pose_num: int, 
            train_env_identity = False, total_env_identity = 2,
            train_grasp_identity = False, 
            train_inside_feasibility = False, 
            binary_grasp = False, 
            open_close_drawer = False, 
            softmax_identity = False, 
            seperate_identity = False, 
            train_obj_boundary = False,
            train_obj_move = False, 
            one_bit_env = False, 
            pe = False,
            transformer_decoder = False,
            remove_orientation = False,
            pose_trans_decoder = False,
        ): 
        super(QuickReadoutNet, self).__init__()
        d_input = width
        
        self.train_env_identity = train_env_identity
        self.train_grasp_identity = train_grasp_identity
        self.train_inside_feasibility = train_inside_feasibility
        self.binary_grasp = binary_grasp

        self.open_close_drawer = open_close_drawer 
        self.softmax_identity = softmax_identity
        self.seperate_identity = seperate_identity
        self.train_obj_boundary = train_obj_boundary
        self.train_obj_move = train_obj_move
        self.pe = pe
        self.transformer_decoder = transformer_decoder
        self.pose_trans_decoder = pose_trans_decoder
        self.remove_orientation = remove_orientation

        total_env_identity = 3

        
        self.one_hot_encoding_dim = (int)(width/2)

        self.one_hot_encoding_embed = nn.Sequential(
                nn.Linear(n_objects, self.one_hot_encoding_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.one_hot_encoding_dim, self.one_hot_encoding_dim)
            )
        
        self.action_emb = nn.Sequential(
                nn.Linear(input_feature_num, width),
                nn.ReLU(),
                nn.Linear(width, width)
            )

        self.env_output_identity = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, total_env_identity),
            nn.Sigmoid()
        ) 

        
        self.grasp_output_identity = self.get_head(d_input, d_hidden, 2, 2)
        
    
        self.pose_estimation = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, pose_num)
        )

        self.n_unary = n_unary
        self.n_binary = n_binary
        self.d_hidden = d_hidden

        self.f_unary = self.get_head_unary(d_input, d_hidden, 1, n_unary)
        self.f_binary = self.get_head(d_input, d_hidden, 2, n_binary)
        self.ln_post = nn.LayerNorm(width)

    def get_head(self, d_input, d_hidden, n_args, n_binary):
        if d_hidden > 1:
            head = nn.Sequential(
                nn.Linear(d_input * n_args, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, n_binary),
                nn.Sigmoid()
            )
        else:
            head = nn.Sequential(
                nn.Linear(d_input * n_args, d_hidden),
                nn.Sigmoid()
            )
        return head

    def get_head_unary(self, d_input, d_hidden, n_args, n_unary):
        if d_hidden > 1:
            head = nn.Sequential(
                nn.Linear(d_input * n_args, d_hidden),
                nn.ReLU(),
                nn.Linear(d_hidden, n_unary)
            )
        else:
            head = nn.Sequential(
                nn.Linear(d_input * n_args, d_hidden)
            )
        return head

    
    def forward(self, objs, edge_index):
        
        batch_size, n_obj = objs.shape[:2] # # shape = [*, n_obj, width]

        x = objs # x = (pointconv feature + one hot encoding(128)) * 

        
        z = self.f_unary(x)

        self.env_identity = self.env_output_identity(x)

        self.predicted_pose = self.pose_estimation(x)

        x1 = x[:, edge_index[0, :], :]

        x2 = x[:, edge_index[1, :], :]

        concat_x = torch.cat([x1, x2], dim=-1)
        y = self.f_binary(concat_x)

        self.binary_grasp_identity = self.grasp_output_identity(concat_x)
        
        return_dict = {'pred': z,
        'current_embed': x,
        'pred_sigmoid': y,
        'env_identity': self.env_identity,
        'grasp_identity': self.binary_grasp_identity,
        'predicted_pose': self.predicted_pose}

        return return_dict
    
        
class PointConv(nn.Module):
    def __init__(self, normal_channel=False, use_rgb=False, output_dim = 128):
        super(PointConv, self).__init__()

        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        
        if use_rgb:
            rgb_channel = 3
        else:
            rgb_channel = 0
        self.output_dim = output_dim
        self.normal_channel = normal_channel
        
        self.sa1 = PointConvDensitySetAbstraction(npoint=128, nsample=8, in_channel=6+additional_channel+rgb_channel, mlp=[32], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=64, nsample=16, in_channel= 32 + 3, mlp=[64], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel= 64 + 3, mlp=[output_dim], bandwidth = 0.4, group_all=True) 

    def forward(self, xyz):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz[:, :3, :]
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        x = l3_points.view(B, self.output_dim)

        return x
