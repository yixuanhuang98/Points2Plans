import numpy as np
import argparse
import pickle
import sys
import os
import pprint
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import copy
import random
import torch.nn.functional as F

sys.path.append(os.getcwd())
from relational_dynamics.utils.colors import bcolors
from relational_dynamics.utils.data_utils import get_norm, get_activation
from relational_dynamics.utils.other_util import LinearBlock, MLP, create_log_dirs
from relational_dynamics.dataloader.dataloader import DataLoader
from relational_dynamics.model.models import PointConv, EmbeddingNetTorch, QuickReadoutNet
from relational_dynamics.utils import torch_util
from relational_dynamics.utils import parse_util
from relational_dynamics.config.base_config import BaseConfig
from itertools import permutations
from torch.utils.tensorboard import SummaryWriter


class RelationalDynamics(object):
    def __init__(self, config):

        self.config = config

        args = config.args
        self.args = config.args
        
        self.timestr = time.strftime("%Y-%m-%d-%H-%M-%S")

        self.params = {
            'theta_predicte_lr_fb_ab' : np.pi / args.relation_angle, # 45 degrees
            'occ_IoU_threshold' : 0.5,
        }

        if self.args.use_tensorboard and len(config.args.checkpoint_path) == 0:
            self.writer = SummaryWriter(log_dir = "../runs/"+self.timestr)
            self.loss_iter = 0



        self.action_list = []
        self.goal_relation_list = []
        self.gt_extents_range_list = []
        self.gt_pose_list = []

        
        self.dataloader = DataLoader(
            config,
            use_multiple_train_dataset = args.use_multiple_train_dataset,
            use_multiple_test_dataset = args.use_multiple_test_dataset, 
            pick_place = args.pick_place, 
            pushing = args.pushing,
            stacking = True, 
            set_max = args.set_max, 
            max_objects = args.max_objects,
            online_planning = args.online_planning,
            start_id = args.start_id, 
            max_size = args.max_size, 
            start_test_id = args.start_test_id, 
            test_max_size = args.test_max_size,
            updated_behavior_params = args.updated_behavior_params,
            save_data_path = args.save_data_path,
            evaluate_new = args.evaluate_new, 
            evaluate_pickplace = args.evaluate_pickplace,
            using_multi_step_statistics = args.using_multi_step_statistics,
            total_multi_steps = args.total_sub_step,
            use_shared_latent_embedding = args.use_shared_latent_embedding,
            use_seperate_latent_embedding = args.use_seperate_latent_embedding,
            push_3_steps = args.push_3_steps,
            POMDP_push = args.POMDP_push, 
            sudo_pickplace = args.sudo_pickplace, 
            push_steps = args.push_steps, 
            single_step_training = args.single_step_training, 
            add_noise_pc = args.add_noise_pc, 
            train_object_identity = args.train_object_identity,
            use_rgb = args.use_rgb,
            use_boundary_relations = args.use_boundary_relations,
            consider_z_offset = args.consider_z_offset,
            seperate_env_id = args.seperate_env_id,
            max_env_num = args.max_env_num,
            env_first_step = args.env_first_step,
            use_discrete_z = args.use_discrete_z,
            fast_training = args.fast_training,
            one_bit_env = args.one_bit_env,
            rcpe = args.rcpe,
            pe = args.pe, 
            relation_angle = args.relation_angle,
            bookshelf_env_shift = args.bookshelf_env_shift,
            lfd_search = args.lfd_search,
            get_hidden_label = args.get_hidden_label,
            get_inside_relations = args.get_inside_relations,
            enable_place_inside = args.enable_place_inside,
            binary_grasp = args.binary_grasp, 
            open_close_drawer = args.open_close_drawer, 
            softmax_identity = args.softmax_identity, 
            train_inside_feasibility = args.train_inside_feasibility, 
            use_discrete_place = args.use_discrete_place, 
            seperate_place = args.seperate_place, 
            enable_leap_num = args.enable_leap_num,
            batch_feasibility = args.batch_feasibility
            )
        
                
           
        args = config.args
        
        self.emb_model = PointConv(normal_channel=False, use_rgb = args.use_rgb, output_dim = args.node_emb_size)


        self.classif_model = EmbeddingNetTorch(n_objects = args.max_objects,
                                width = args.node_emb_size*2, 
                                layers = args.n_layers, 
                                heads = args.n_heads, 
                                input_feature_num = args.max_objects + 3,
                                d_hidden = args.d_hidden, 
                                n_unary = args.max_objects + 3, 
                                n_binary = args.z_dim,
                                simple_encoding = args.simple_encoding,
                                transformer_dynamics = args.transformer_dynamics,
                                seperate_discrete_continuous = args.seperate_discrete_continuous,
                                torch_embedding = args.torch_embedding,
                                complicated_pre_dynamics = args.complicated_pre_dynamics,
                                direct_transformer = args.direct_transformer,
                                enable_high_push = args.enable_high_push,
                                enable_place_inside = args.enable_place_inside, 
                                use_discrete_place = args.use_discrete_place,
                                latent_discrete_continuous = args.latent_discrete_continuous, 
                                seperate_place = args.seperate_place,
                                use_seperate_latent_embedding = args.use_seperate_latent_embedding,
                                seperate_action_emb = args.seperate_action_emb,
                                dim_feedforward = args.dim_feedforward,
                                use_mlp_encoder = args.use_mlp_encoder, 
                                )
        
        self.classif_model_decoder = QuickReadoutNet(n_objects = args.max_objects,
                            width = args.node_emb_size*2, 
                            layers = args.n_layers, 
                            heads = args.n_heads, 
                            input_feature_num = args.max_objects + 3,
                            d_hidden = args.d_hidden, 
                            n_unary = args.max_objects + 3, 
                            n_binary = args.z_dim,
                            pose_num = args.pose_num, 
                            train_env_identity = args.train_env_identity,
                            train_grasp_identity = args.train_grasp_identity, 
                            train_inside_feasibility = args.train_inside_feasibility, 
                            binary_grasp = args.binary_grasp, 
                            open_close_drawer = args.open_close_drawer, 
                            softmax_identity = args.softmax_identity, 
                            seperate_identity = args.seperate_identity, 
                            one_bit_env = args.one_bit_env,
                            pe = args.pe,
                            train_obj_move = args.train_obj_move,
                            train_obj_boundary = args.train_obj_boundary, 
                            transformer_decoder = args.transformer_decoder,
                            remove_orientation = args.remove_orientation,
                            pose_trans_decoder = args.pose_trans_decoder,
                            dim_feedforward = args.dim_feedforward)
    

        

        self.opt_emb = optim.Adam(self.emb_model.parameters(), lr=args.emb_lr)
        
        self.opt_classif = optim.Adam(self.classif_model.parameters(), lr=args.learning_rate) 
        self.opt_classif_decoder = optim.Adam(self.classif_model_decoder.parameters(), lr=args.learning_rate) 
        

        self.dynamics_loss = nn.MSELoss()
        self.sum_dyna_loss = nn.L1Loss()
        
        self.bce_loss = nn.BCELoss()

    def mySqrtLoss (self, distance):
        return  self.args.sqrt_var * torch.sqrt(12*distance)
       
    def get_model_list(self):
        return [self.emb_model ,self.classif_model,self.classif_model_decoder]
        
    def get_state_dict(self):
        return {
            'emb_model': self.emb_model.state_dict(),
            'classif_model': self.classif_model.state_dict(),
            'classif_model_decoder': self.classif_model_decoder.state_dict(),
        }

    def set_model_device(self, device=torch.device("cpu")):
        model_list = self.get_model_list()
        for m in model_list:
            m.to(device)
    
    def model_checkpoint_dir(self):
        '''Return the directory to save models in.'''
        return self.config.get_model_checkpoint_dir()

    def model_checkpoint_filename(self, epoch):
        return os.path.join(self.model_checkpoint_dir(),
                            'cp_{}.pth'.format(epoch))

    def save_checkpoint(self, epoch):
        cp_filepath = self.model_checkpoint_filename(epoch)
        torch.save(self.get_state_dict(), cp_filepath)
        print(bcolors.c_red("Save checkpoint: {}".format(cp_filepath)))

    def load_checkpoint(self, checkpoint_path):
        cp_models = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        self.emb_model.load_state_dict(cp_models['emb_model'])
        self.classif_model.load_state_dict(cp_models['classif_model'])
        self.classif_model_decoder.load_state_dict(cp_models['classif_model_decoder'])

    def process_data(self, batch_data, push_3_steps = False): 
        '''Process raw batch data and collect relevant objects in a dict.'''
        args = self.config.args
        x_dict = dict()

        
        # assert(len(batch_data) == 1)

        x_dict['batch_num_objects'] = []
        x_dict['batch_action'] = []
        x_dict['batch_all_obj_pair_relation'] = []
        x_dict['batch_one_hot_encoding'] = []
        x_dict['batch_skill_label'] = []
        x_dict['batch_voxel_list_single'] = []
        x_dict['batch_env_identity'] = []
        x_dict['batch_grasp_identity'] = []
        x_dict['batch_6DOF_pose'] = []
        x_dict['support_suface_id'] = []
        x_dict['buffer_tensor_0'] = []

        for b, data in enumerate(batch_data):
            x_dict['batch_num_objects'].append(data['num_objects'])
            x_dict['batch_action'].append(data['action'])
            x_dict['batch_all_obj_pair_relation'].append(data['relation'])
            x_dict['batch_one_hot_encoding'].append(data['one_hot_encoding'])
            x_dict['batch_skill_label'].append(data['all_action_label'])
            x_dict['batch_voxel_list_single'].append(data['all_object_pair_voxels_single'])
            x_dict['batch_env_identity'].append(data['env_identity'])
            x_dict['batch_grasp_identity'].append(data['all_gt_grapable_list'])
            x_dict['batch_6DOF_pose'].append(data['all_6DOF_pose_fast'])
            x_dict['batch_edge_attr'] = data['edge_attr']
            x_dict['batch_obj_boundary'] = data['all_obj_boundaty']
            x_dict['batch_position'] = data['position']
            x_dict['batch_quaternian'] = data['quaternian']
            x_dict['batch_extents'] = data['extents']
            x_dict['batch_all_hidden_tensor'] = data['all_hidden_tensor']
            x_dict['new_latent'] = data['new_latent']
            x_dict['support_suface_id'].append(data['support_suface_id'])
            x_dict['buffer_tensor_0'].append(data['buffer_tensor_0'])
            x_dict['batch_all_hidden_label'] = data['all_hidden_label']
            
        x_dict['batch_env_identity'] = torch.squeeze(torch.stack(x_dict['batch_env_identity']), 1)
        x_dict['batch_all_obj_pair_relation'] = torch.squeeze(torch.stack(x_dict['batch_all_obj_pair_relation']), 1)
        x_dict['batch_voxel_list_single'] = torch.squeeze(torch.stack(x_dict['batch_voxel_list_single']), 1)
        x_dict['batch_action'] = torch.squeeze(torch.stack(x_dict['batch_action']), 1)
        x_dict['batch_skill_label'] = np.squeeze(np.stack(x_dict['batch_skill_label']), 1)
        x_dict['batch_one_hot_encoding'] = torch.squeeze(torch.stack(x_dict['batch_one_hot_encoding']), 1)
        x_dict['batch_grasp_identity'] = torch.squeeze(torch.stack(x_dict['batch_grasp_identity']), 1)
        x_dict['batch_6DOF_pose'] = torch.squeeze(torch.stack(x_dict['batch_6DOF_pose']), 1)
        
        return x_dict

    def process_data_plan(self, batch_data, push_3_steps = False): 
        '''Process raw batch data and collect relevant objects in a dict.'''
        args = self.config.args
        x_dict = dict()

        
        assert(len(batch_data) == 1)

        for b, data in enumerate(batch_data):
            x_dict['batch_num_objects'] = data['num_objects']
            x_dict['batch_action'] = data['action']
            x_dict['batch_all_obj_pair_relation'] = data['relation']
            x_dict['batch_one_hot_encoding'] = data['one_hot_encoding']
            x_dict['batch_edge_attr'] = data['edge_attr']
            x_dict['batch_skill_label'] = data['all_action_label']
            x_dict['batch_voxel_list_single'] = data['all_object_pair_voxels_single']
            x_dict['batch_env_identity'] = data['env_identity']
            x_dict['batch_grasp_identity'] = data['all_gt_grapable_list']
            x_dict['batch_6DOF_pose'] = data['all_6DOF_pose_fast']
            x_dict['batch_obj_boundary'] = data['all_obj_boundaty']

            x_dict['batch_position'] = data['position']
            x_dict['batch_quaternian'] = data['quaternian']
            x_dict['batch_extents'] = data['extents']
            x_dict['batch_all_hidden_tensor'] = data['all_hidden_tensor']
            x_dict['new_latent'] = data['new_latent']

            x_dict['support_suface_id'] = data['support_suface_id']
            x_dict['buffer_tensor_0'] = data['buffer_tensor_0']

            x_dict['batch_all_hidden_label'] = data['all_hidden_label']

            
        return x_dict

    def training(self,
                x_tensor_dict,
                x_tensor_dict_next,
                batch_size,
                train=False,
                threshold = 0):

        batch_result_dict = {}
        device = self.config.get_device()

        total_steps = x_tensor_dict['batch_all_obj_pair_relation'].shape[1] 

        self.num_nodes = x_tensor_dict['batch_num_objects'] 

        current_latent_list = []
        current_output_classifier_list = []
        current_output_pose_list = []
        current_env_identity_list = []
        current_grasp_identity_list = []
        
        
        x_tensor_dict['batch_all_obj_pair_relation'] = x_tensor_dict['batch_all_obj_pair_relation'] 

        self.edge_index = x_tensor_dict['batch_edge_attr']

        x_tensor_dict['batch_all_obj_pair_relation'] = torch.transpose(x_tensor_dict['batch_all_obj_pair_relation'], 0, 1)
        
        x_tensor_dict['batch_grasp_identity'] = torch.transpose(x_tensor_dict['batch_grasp_identity'], 0, 1)
        
        x_tensor_dict['batch_6DOF_pose'] = torch.transpose(x_tensor_dict['batch_6DOF_pose'], 0, 1)
        x_tensor_dict['batch_6DOF_pose'] = x_tensor_dict['batch_6DOF_pose'][:, :, :, :2]
        
        x_tensor_dict['batch_env_identity'] = torch.transpose(x_tensor_dict['batch_env_identity'], 0, 1)
        
        self.src_key_padding_mask = (x_tensor_dict['batch_env_identity'][:, :, :, 0]==-1)
        
        for each_i in range(self.src_key_padding_mask.shape[0]):
            for each_j in range(self.src_key_padding_mask.shape[1]):
                if torch.all(self.src_key_padding_mask[each_i][each_j]):
                    self.src_key_padding_mask[each_i][each_j][0] = False

        
        self.dynamic_src_padding_mask = torch.zeros(self.src_key_padding_mask.shape[0], self.src_key_padding_mask.shape[1], self.src_key_padding_mask.shape[2] + 2).to(device)
        self.dynamic_src_padding_mask[:,:,:-2] = self.src_key_padding_mask


        for this_step in range(total_steps):
            voxel_data_single = x_tensor_dict['batch_voxel_list_single'][:, this_step, :, : ,:]
            reshaped_voxel_data_single = voxel_data_single.reshape(voxel_data_single.shape[0]*voxel_data_single.shape[1], voxel_data_single.shape[2], voxel_data_single.shape[3])
            img_emb_single = self.emb_model(reshaped_voxel_data_single)
            img_emb_single = img_emb_single.reshape(voxel_data_single.shape[0], voxel_data_single.shape[1], img_emb_single.shape[1])
            

            one_hot_encoding_tensor = x_tensor_dict['batch_one_hot_encoding'] 
            latent_one_hot_encoding = self.classif_model.one_hot_encoding_embed(torch.argmax(one_hot_encoding_tensor, dim = 2))
            
            node_pose = torch.cat([img_emb_single, latent_one_hot_encoding], dim = -1)

            
            outs_decoder = self.classif_model_decoder(node_pose, self.edge_index)

            current_latent_list.append([node_pose])

            current_output_classifier_list.append(outs_decoder['pred_sigmoid'][:])

            current_output_pose_list.append(outs_decoder['predicted_pose'][:])
            
            current_env_identity_list.append(outs_decoder['env_identity'][:]) 

            current_grasp_identity_list.append(outs_decoder['grasp_identity'][:]) 

        seq = 0 
        
        action_torch = x_tensor_dict['batch_action'][:, seq]
        skill_label = x_tensor_dict['batch_skill_label'][:, seq]
            
        current_latent = current_latent_list[0][0]
        
        discrete_action = self.classif_model.one_hot_encoding_embed(torch.argmax(action_torch[:, 0, 1:-3], dim = -1))
        discrete_action = discrete_action.view(discrete_action.shape[0], 1, discrete_action.shape[1])

        continuous_action = []
        for batch_id in range(discrete_action.shape[0]):
            if skill_label[batch_id] == 0:
                continuous_action.append(self.classif_model.continuous_action_emb(action_torch[batch_id, 0, -3:-1])) 
            elif skill_label[batch_id] == 1:
                continuous_action.append(self.classif_model.continuous_action_emb_1(action_torch[batch_id, 0, -3:-1]))
        continuous_action = torch.stack(continuous_action)
        continuous_action = continuous_action.view(continuous_action.shape[0], 1, continuous_action.shape[1])
        
        
        current_action_continuous = torch.cat((discrete_action, continuous_action), axis = -1)

        discrete_place_id_tensor = []
        for batch_id in range(discrete_action.shape[0]):
            if x_tensor_dict['support_suface_id'][batch_id][seq].shape[0] == 1 and x_tensor_dict['support_suface_id'][batch_id][seq].shape[1] >= 1:
                assert x_tensor_dict['support_suface_id'][batch_id][seq][0][0] == x_tensor_dict['support_suface_id'][batch_id][seq][0][1]
                discrete_place_id = x_tensor_dict['support_suface_id'][batch_id][seq][0][0]
                discrete_place_id_tensor.append(self.classif_model.one_hot_encoding_embed(discrete_place_id))
            else:  
                discrete_place_id_tensor.append(x_tensor_dict['buffer_tensor_0'][batch_id][seq][0])
        
        discrete_place_id_tensor = torch.stack(discrete_place_id_tensor)
        discrete_place_id_tensor = discrete_place_id_tensor.view(discrete_place_id_tensor.shape[0], 1, discrete_place_id_tensor.shape[1])
        
        current_action = torch.cat((discrete_place_id_tensor, continuous_action), axis = -1)


        graph_node_action = torch.cat((current_latent, current_action_continuous, current_action), axis = 1)

        
        current_latent = []
        for batch_id in range(discrete_action.shape[0]):
            if skill_label[batch_id] == 0:
                current_latent.append(self.classif_model.graph_dynamics_0(graph_node_action[batch_id], src_key_padding_mask = self.dynamic_src_padding_mask[0][batch_id]))
            elif skill_label[batch_id] == 1:
                current_latent.append(self.classif_model.graph_dynamics_1(graph_node_action[batch_id], src_key_padding_mask = self.dynamic_src_padding_mask[0][batch_id]))
        current_latent = torch.stack(current_latent)
        
        if self.config.args.delta_forward:
            delta_latent = current_latent[:, :-2, :]
            pred_latent = delta_latent + current_latent_list[0][0] 
        else:
            pred_latent = current_latent[:, :-2, :]
        
        outs_decoder_2_edge = self.classif_model_decoder(pred_latent, self.edge_index) 
        predicted_predicates = outs_decoder_2_edge['pred_sigmoid'][:]

        if self.config.args.delta_forward:
            outs_decoder_2_edge_delta = self.classif_model_decoder(delta_latent, self.edge_index)
            predicted_pose = outs_decoder_2_edge_delta['predicted_pose'][:]
        else:
            predicted_pose = outs_decoder_2_edge['predicted_pose'][:]
        predicted_env = outs_decoder_2_edge['env_identity'][:]
        predicted_feasibility = outs_decoder_2_edge['grasp_identity'][:]

        total_loss = 0

        relational_mask = (x_tensor_dict['batch_all_obj_pair_relation']==-1)
        
        env_mask = (x_tensor_dict['batch_env_identity']==-1)

        object_level_mask = env_mask[:, :, :, [0]]

        latent_space_mask = object_level_mask.repeat(1,1,1,256)
        

        graspable_mask = (x_tensor_dict['batch_grasp_identity']==-1)
        position_mask = (x_tensor_dict['batch_6DOF_pose']==-1)
        
        for i in range(len(current_output_classifier_list)):            
            total_loss += self.bce_loss(current_output_classifier_list[i][~relational_mask[i]], x_tensor_dict['batch_all_obj_pair_relation'][i][~relational_mask[i]])
            total_loss += self.bce_loss(current_env_identity_list[i][~env_mask[i]], x_tensor_dict['batch_env_identity'][i][~env_mask[i]])
            total_loss += self.bce_loss(current_grasp_identity_list[i][~graspable_mask[i]], x_tensor_dict['batch_grasp_identity'][i][~graspable_mask[i]])

        total_loss += self.dynamics_loss(pred_latent[~latent_space_mask[1]], current_latent_list[1][0][~latent_space_mask[1]])
        total_loss += self.mySqrtLoss(self.sum_dyna_loss(predicted_pose[~position_mask[1]] + x_tensor_dict['batch_6DOF_pose'][0][~position_mask[1]], x_tensor_dict['batch_6DOF_pose'][1][~position_mask[1]]))
        total_loss += self.bce_loss(predicted_predicates[~relational_mask[1]], x_tensor_dict['batch_all_obj_pair_relation'][1][~relational_mask[1]])  
        total_loss += self.bce_loss(predicted_env[~env_mask[i]], x_tensor_dict['batch_env_identity'][1][~env_mask[i]])
        total_loss += self.bce_loss(predicted_feasibility[~graspable_mask[1]], x_tensor_dict['batch_grasp_identity'][1][~graspable_mask[1]])

        with torch.autograd.set_detect_anomaly(True):
            self.opt_emb.zero_grad()
            self.opt_classif.zero_grad()
            self.opt_classif_decoder.zero_grad()
            total_loss.backward()
            self.opt_emb.step()
            self.opt_classif.step()
            self.opt_classif_decoder.step()
    
        return batch_result_dict

    def collision_check_2D(self, bounding_box_1, bounding_box_2):
        bbox1_left = bounding_box_1[0][0]
        bbox1_right = bounding_box_1[2][0]
        bbox1_top = bounding_box_1[0][1]
        bbox1_bottom = bounding_box_1[1][1]

        bbox2_left = bounding_box_2[0][0]
        bbox2_right = bounding_box_2[2][0]
        bbox2_top = bounding_box_2[0][1]
        bbox2_bottom = bounding_box_2[1][1]

        if (bbox1_left < bbox2_right and
            bbox1_right > bbox2_left and
            bbox1_top < bbox2_bottom and
            bbox1_bottom > bbox2_top):
            return True
        else:
            return False
    
    def planner(self):
        device = self.device
        action_selections = self.args.planning_batch_size
        
        obj_select_range = []
        
        for each_env_id in range(self.env_identity.shape[0]):
            if self.env_identity[each_env_id, 0] > 0.5 or self.env_identity[each_env_id, 1] > 0.5 or self.env_identity[each_env_id, 2] > 0.5:
                obj_select_range.append(each_env_id)

        if self.num_nodes == 5:
            # cached task plans for packing three objects based on the output of LLMs.
            self.task_planner = [[0, 0, 3], [0, 1, 3], [0, 2, 3]]
        elif self.num_nodes == 6:
            # cached task plans for packing four objects based on the output of LLMs.
            self.task_planner = [[0, 0, 4], [0, 1, 4], [0, 2, 4], [0, 3, 4]]   
        elif self.num_nodes == 7:
            # cached task plans for packing five objects based on the output of LLMs.
            self.task_planner = [[0, 0, 5], [0, 1, 5], [0, 2, 5], [0, 3, 5], [0, 4, 5]]
        
        start_time = time.time()

        with torch.no_grad():
            costs = []
            previous_pc_copy = copy.deepcopy(self.previous_pc)
            previous_pc_center_copy = copy.deepcopy(self.pc_center_tensor)

            previous_pc_copy = previous_pc_copy.view(1, previous_pc_copy.shape[0], previous_pc_copy.shape[1], previous_pc_copy.shape[2])

            previous_pc_copy = previous_pc_copy.repeat(action_selections, 1, 1, 1)

            previous_pc_center_copy = previous_pc_center_copy.view(1, previous_pc_center_copy.shape[0], previous_pc_center_copy.shape[1])

            previous_pc_center_copy = previous_pc_center_copy.repeat(action_selections, 1, 1)

            one_hot_encoding_tensor = self.x_tensor_dict['batch_one_hot_encoding'] 
            latent_one_hot_encoding = self.classif_model.one_hot_encoding_embed(torch.argmax(one_hot_encoding_tensor, dim = 1))
            
            
            latent_one_hot_encoding = torch.unsqueeze(latent_one_hot_encoding, 0)
            latent_one_hot_encoding = latent_one_hot_encoding.repeat(action_selections, 1, 1)

            
            ori_this_time_step_embed = self.this_time_step_embed
            
            
            for planning_iter in range(10):
                self.this_time_step_embed = ori_this_time_step_embed
                all_action_tensor = []
                self.previous_pc = copy.deepcopy(previous_pc_copy)
                self.pc_center_tensor = copy.deepcopy(previous_pc_center_copy)
                previous_pc_center_numpy = previous_pc_center_copy.detach().cpu().numpy()
                
                self.all_min_identity = []
                for shoot_i in range(len(self.task_planner)):
                    this_sequence = []

                    feasibility_leap = 1
                    for j in range(action_selections):
                        current_task_plan = self.task_planner[shoot_i]
                        obj_mov = current_task_plan[1]
                        if self.args.use_discrete_place:
                            place_id = current_task_plan[2] 
                        skill_iter = current_task_plan[0] 
                        action_numpy = np.zeros((self.num_nodes, 3))
                        
                        action_numpy[obj_mov][0] = np.random.uniform(previous_pc_center_numpy[j][place_id][0] - previous_pc_center_numpy[j][obj_mov][0] - self.args.x_range/2, previous_pc_center_numpy[j][place_id][0] - previous_pc_center_numpy[j][obj_mov][0] + self.args.x_range/2) 
                        action_numpy[obj_mov][1] = np.random.uniform(previous_pc_center_numpy[j][place_id][1] - previous_pc_center_numpy[j][obj_mov][1] - self.args.y_range/2, previous_pc_center_numpy[j][place_id][1] - previous_pc_center_numpy[j][obj_mov][1] + self.args.y_range/2)
                        
                        if current_task_plan[1] != current_task_plan[2]:
                            action_numpy[obj_mov][2] = previous_pc_center_numpy[j][place_id][2]

                        if self.args.use_seperate_latent_embedding or self.args.use_shared_latent_embedding:
                            action = np.zeros((1, self.args.max_objects + 3 + 1))
                            
                            for i in range(action.shape[0]):
                                action[i][0] = skill_iter
                                action[i][obj_mov + 1] = 1
                                action[i][-3:] = action_numpy[obj_mov]
                        
                        sample_action = torch.Tensor(action).to(device)
                        

                        this_sequence.append(sample_action)
                    
                    tensor_action = torch.stack(this_sequence)
                    tensor_action = tensor_action.view(tensor_action.shape[0], tensor_action.shape[2])
                    all_action_tensor.append(tensor_action)

                    seq = shoot_i
                    current_task_plan = self.task_planner[seq]
                    skill_iter = current_task_plan[0]
                    obj_mov = current_task_plan[1]
                    place_id = current_task_plan[2]
                    
                    if skill_iter == 0 and current_task_plan[1] == current_task_plan[2]:
                        place_id = current_task_plan[1]
                        self.place_height = torch.zeros(previous_pc_center_numpy.shape[0], previous_pc_center_numpy.shape[1]).to(device)
                        self.place_height[:, :] = previous_pc_center_numpy[:, obj_mov, 2]
                    else:
                        place_id = current_task_plan[2] 
                        self.place_height = previous_pc_center_numpy[:, place_id, 2] + previous_pc_center_numpy[:, obj_mov, 2]
                        self.place_height = torch.Tensor(self.place_height).to(device)
                        self.place_height = self.place_height.unsqueeze(-1).repeat(1, previous_pc_center_copy.shape[1])
                        
                    
                    if seq == 0:
                        self.this_time_step_embed = self.this_time_step_embed.repeat(action_selections, 1, 1)

                    current_latent = self.this_time_step_embed

                    
                    if True:
                        if self.args.seperate_discrete_continuous:
                            discrete_action = self.classif_model.one_hot_encoding_embed(torch.argmax(all_action_tensor[seq][:, 1:-3], dim = 1))
                            
                            if skill_iter == 0:
                                continuous_action = self.classif_model.continuous_action_emb(all_action_tensor[seq][:, -3:-1])
                            elif skill_iter == 1:
                                continuous_action = self.classif_model.continuous_action_emb_1(all_action_tensor[seq][:, -3:-1])
                        
                            current_action_continuous = torch.cat((discrete_action, continuous_action), axis = -1)
                            if place_id == obj_mov:
                                discrete_place_id_tensor = torch.zeros(discrete_action.shape[0], discrete_action.shape[1]).to(device)
                                
                                current_action = torch.cat((discrete_place_id_tensor, continuous_action), axis = -1)
                            else:
                                discrete_place_id_numpy = np.zeros((discrete_action.shape[0], ))
                                discrete_place_id_numpy[:] = place_id
                                discrete_place_id = torch.Tensor(discrete_place_id_numpy).type(torch.int64).to(device)
                                discrete_place_id_tensor = self.classif_model.one_hot_encoding_embed(discrete_place_id)
                                current_action = torch.cat((discrete_place_id_tensor, continuous_action), axis = -1)

                    current_action = current_action.view(current_action.shape[0], 1, current_action.shape[1])
                    current_action_continuous = current_action_continuous.view(current_action_continuous.shape[0], 1, current_action_continuous.shape[1])

                    graph_node_action = torch.cat((current_latent, current_action_continuous, current_action), axis = 1)

                    if skill_iter == 0:
                        current_latent = self.classif_model.graph_dynamics_0(graph_node_action)
                    elif skill_iter == 1:
                        current_latent = self.classif_model.graph_dynamics_1(graph_node_action)
                    
                    if self.config.args.delta_forward:
                        delta_latent = current_latent[:, :-2, :]
                        current_latent = delta_latent + self.this_time_step_embed
                    else:
                        current_latent = current_latent[:, :-2, :]
                    
                    obj_mov_id = obj_mov
                    if self.config.args.delta_forward:
                        delta_change_torch = self.classif_model_decoder.pose_estimation(delta_latent)
                        change_id_torch = torch.where(torch.sum(torch.abs(delta_change_torch), axis = 2) > self.args.delta_threshold)
                    else:
                        delta_change_torch = self.classif_model_decoder.pose_estimation(current_latent)
                        previous_xy_center_torch = torch.Tensor(previous_pc_center_numpy[:, :, :2]).to(device)
                        change_id_torch = torch.where(torch.sum(torch.abs(delta_change_torch - previous_xy_center_torch), axis = 2) > self.args.delta_threshold)
                    
                    z_height_change = torch.zeros(delta_change_torch.shape[0], delta_change_torch.shape[1]).to(device)
                    original_height_tensor = torch.Tensor(previous_pc_center_numpy[:, :, 2]).to(device)
                    
                    if skill_iter == 0:
                        z_height_change[change_id_torch] = self.place_height[change_id_torch] - original_height_tensor[change_id_torch]
                        
                    z_height_change = z_height_change.unsqueeze(-1)

                    all_change = torch.cat((z_height_change, delta_change_torch), axis = 2)
                    
                    all_change_flatten = all_change.unsqueeze(-1).repeat(1, 1, 1, self.x_tensor_dict['batch_voxel_list_single'][0].shape[2])

                    self.previous_pc[:, obj_select_range, :, :] = self.previous_pc[:, obj_select_range, :, :] + all_change_flatten[:, obj_select_range, :, :]

                    new_mean = torch.mean(self.previous_pc, axis = 3).detach().cpu().numpy()
                    
                    
                    if self.config.args.latent_forward:
                        self.this_time_step_embed = current_latent
                    else:
                        
                        img_emb_single_list = []
                        for each_pc_id in range(self.previous_pc.shape[1]):
                            if each_pc_id == 0:
                                img_emb_single = self.emb_model(self.previous_pc[:, each_pc_id, :, :])
                                img_emb_single = torch.unsqueeze(img_emb_single, 1)
                            else:
                                last_step = self.emb_model(self.previous_pc[:, each_pc_id, :, :])
                                last_step = torch.unsqueeze(last_step, 1)
                                img_emb_single = torch.cat((img_emb_single, last_step), dim = 1) 
                            
                        node_pose = torch.cat([img_emb_single, latent_one_hot_encoding], dim = 2)

                        self.this_time_step_embed = node_pose

                    x = self.this_time_step_embed

                    x1 = x[:, self.edge_index[0, :], :]

                    x2 = x[:, self.edge_index[1, :], :]

                    concat_x = torch.cat([x1, x2], dim=-1)

                    self.min_identity = self.classif_model_decoder.grasp_output_identity(concat_x)
                    
                    self.collision_arrs = []

                    self.all_min_identity.append(self.min_identity)

                for each_sample_id in range(action_selections):
                    feasibility_leap = 1
                    for seq in range(len(self.task_planner)):
                        self.min_identity = self.all_min_identity[seq][each_sample_id]

                        current_task_plan = self.task_planner[seq]
                        for each_min_identity in range(current_task_plan[1]*(self.previous_pc.shape[1] - 1), (current_task_plan[1] + 1)*(self.previous_pc.shape[1] - 1)):
                            if self.min_identity[each_min_identity][1] > 0.5:
                                feasibility_leap = 0 
                            
                        if seq == len(self.task_planner) - 1 and feasibility_leap == 1:
                            self.collision_arrs = []
                            self.collision_boundary = []

                            for each_pc_i in range(self.previous_pc.shape[1]):
                                if each_pc_i in obj_select_range:
                                    if torch.mean(self.previous_pc[each_sample_id][each_pc_i], axis = 1)[0] > previous_pc_center_numpy[each_sample_id][current_task_plan[2]][2]:
                                        self.collision_arrs.append(torch.mean(self.previous_pc[each_sample_id][each_pc_i], axis = 1).detach().cpu().numpy())
                                        boundary_temp = []
                                        for _ in range(2):
                                            for i_ in range(2):
                                                boundary_temp.append([self.collision_arrs[-1][1] + self.args.x_collision * (_ - 0.5), self.collision_arrs[-1][2] + self.args.y_collision * (i_ - 0.5)])

                                        self.collision_boundary.append(boundary_temp)
                                    else:
                                        feasibility_leap = 0
                                
                            for each_collision_i in range(len(self.collision_boundary) - 1):
                                for each_collision_j in range(each_collision_i + 1, len(self.collision_boundary)):
                                    if self.collision_check_2D(self.collision_boundary[each_collision_i], self.collision_boundary[each_collision_j]) or self.collision_check_2D(self.collision_boundary[each_collision_j], self.collision_boundary[each_collision_i]):
                                        feasibility_leap = 0

                            if feasibility_leap == 1: ## ground 
                                self.collision_arrs = []
                                self.collision_boundary = []
                                for each_pc_i in range(self.previous_pc.shape[1]):
                                    if each_pc_i in obj_select_range:
                                        if torch.mean(self.previous_pc[each_sample_id][each_pc_i], axis = 1)[0] < previous_pc_center_numpy[each_sample_id][current_task_plan[2]][2]:
                                            self.collision_arrs.append(torch.mean(self.previous_pc[each_sample_id][each_pc_i], axis = 1).detach().cpu().numpy())
                                            boundary_temp = []
                                            for _ in range(2):
                                                for i_ in range(2):
                                                    boundary_temp.append([self.collision_arrs[-1][1] + self.args.x_collision * (_ - 0.5), self.collision_arrs[-1][2] + self.args.y_collision * (i_ - 0.5)])

                                            self.collision_boundary.append(boundary_temp)
                                            
                                for each_collision_i in range(len(self.collision_boundary) - 1):
                                    for each_collision_j in range(each_collision_i + 1, len(self.collision_boundary)):
                                        if self.collision_check_2D(self.collision_boundary[each_collision_i], self.collision_boundary[each_collision_j]) or self.collision_check_2D(self.collision_boundary[each_collision_j], self.collision_boundary[each_collision_i]):
                                            feasibility_leap = 0
                    
                    
                    if feasibility_leap == 1:
                        
                        self.min_action = []
                        self.original_pc_numpy = previous_pc_center_copy[0].detach().cpu().numpy()
                        print('original object/env position (xyz):')
                        for _ in range(self.original_pc_numpy.shape[0] - 2):
                            print(f"object_{_ + 1} {self.original_pc_numpy[_]}")
                        for _ in range(self.original_pc_numpy.shape[0] - 2, self.original_pc_numpy.shape[0]):
                            print(f"cupboard horizon layer {_ + 1 - self.original_pc_numpy.shape[0] + 2} {self.original_pc_numpy[_]}")
                        
                        print('returned parameterized primitives:')
                        for _ in range(len(all_action_tensor)):
                            if self.task_planner[_][0] == 0:
                                skill_string = 'pick_place'
                            elif self.task_planner[_][0] == 1:
                                skill_string = 'push/pull'
                            continuous_params = previous_pc_center_copy[0][_, :3] + all_action_tensor[_][each_sample_id][-3:]
                            continuous_params = continuous_params.detach().cpu().numpy()
                            print(f"{skill_string} object_{self.task_planner[_][1] + 1} to {continuous_params}")
                        break

                if each_sample_id == action_selections - 1:
                    print('No feasible solution found this iteration. Continue sampling.')
                else:
                    break
                    
    def planning(self,
                x_tensor_dict,
                x_tensor_dict_next,
                batch_size,
                train=False,
                threshold = 0):
        
        batch_result_dict = {}
        self.device = self.config.get_device()
        device = self.config.get_device()
        args = self.config.args

        total_loss = 0

        self.num_nodes = x_tensor_dict['batch_num_objects'] 

        self.x_tensor_dict = x_tensor_dict
        
        current_grasp_identity_list = []

        current_observation_id = self.args.current_observation_id
        
        voxel_data_single = x_tensor_dict['batch_voxel_list_single'][0]
        img_emb_single = self.emb_model(voxel_data_single)
        one_hot_encoding_tensor = x_tensor_dict['batch_one_hot_encoding'] 
        latent_one_hot_encoding = self.classif_model.one_hot_encoding_embed(torch.argmax(one_hot_encoding_tensor, dim = 1))
        node_pose = torch.cat([img_emb_single, latent_one_hot_encoding], dim = 1)
        if node_pose.shape[0] != 1: 
            node_pose = node_pose.view(1, node_pose.shape[0], node_pose.shape[1])
        
        self.pc_center_tensor = copy.deepcopy(x_tensor_dict['batch_6DOF_pose'][self.args.current_observation_id])
        self.previous_pc = x_tensor_dict['batch_voxel_list_single'][self.args.current_observation_id]

        current_goal_relations = np.zeros((x_tensor_dict['batch_6DOF_pose'][self.args.current_observation_id].shape[0]*(x_tensor_dict['batch_6DOF_pose'][self.args.current_observation_id].shape[0] - 1), self.args.z_dim))

        if self.num_nodes == 5:
            # cached goal predicates for packing three objects based on the output of LLMs.
            # all three objects above the cupboard
            index_i = [2, 6, 10]
            index_j = [5, 5, 5]
            value   = [1, 1, 1]
        elif self.num_nodes == 6:
            # cached goal predicates for packing four objects based on the output of LLMs.
            # all four objects above the cupboard
            index_i = [3, 8, 13, 18]
            index_j = [5, 5, 5, 5]
            value   = [1, 1, 1, 1]
        elif self.num_nodes == 7:
            # cached goal predicates for packing five objects based on the output of LLMs.
            # all five objects above the cupboard
            index_i = [4, 10, 16, 22, 28]
            index_j = [5, 5, 5, 5, 5]
            value   = [1, 1, 1, 1, 1]

    
        for each_i in range(len(index_i)):
            current_goal_relations[index_i[each_i]][index_j[each_i]] = value[each_i]
        
        x_tensor_dict_next['batch_all_obj_pair_relation'] = torch.Tensor(current_goal_relations).to(device)
        
        self.this_time_step_embed = node_pose
        
        nodes = list(range(self.this_time_step_embed.shape[1]))
        edges = list(permutations(nodes, 2))
        self.edge_index = torch.LongTensor(np.array(edges).T)
        
        outs_decoder_final = self.classif_model_decoder(self.this_time_step_embed,  self.edge_index)

    
        self.index_i = index_i 
        self.index_j = index_j
        self.x_tensor_dict_next = x_tensor_dict_next

        if self.args.train_env_identity:
            self.env_identity = outs_decoder_final['env_identity'].cpu().detach().numpy()[0]
        if self.args.train_inside_feasibility:
            self.current_grasp_identity = outs_decoder_final['grasp_identity'].detach().cpu().numpy()[0]

        self.current_relation_all = outs_decoder_final['pred_sigmoid'].detach().cpu().numpy()

        self.planner() 
        

        return batch_result_dict

    def get_next_data_from_dataloader(self, dataloader, train):
        args = self.config.args
        data = None
        data, data_next = dataloader.get_next_all_object_pairs_for_scene(train)
        
        return data, data_next
   
    def RD(self, train=True, threshold = 0.8):
        args = self.config.args
        dataloader = self.dataloader
        device = self.config.get_device()

        train_data_size = dataloader.number_of_scene_data(train)

        self.set_model_device(device)

        num_epochs = args.num_epochs if train else 1

        if train_data_size == 0:
            raise ValueError("Training total size == 0")
        else:
            dataloader.put_all_data_device(device)
            
            for e in range(num_epochs):
                dataloader.reset_scene_batch_sampler(train=train, shuffle=train)

                batch_size = args.batch_size 
                num_batches = train_data_size // batch_size
                if train_data_size % batch_size != 0:
                    num_batches += 1

                
                data_idx = 0


                for batch_idx in range(num_batches):

                    batch_data = []
                    batch_data_next = []

                    while len(batch_data) < batch_size and data_idx < train_data_size:  # in current version, we totally ignore batch size
                        data, data_next = self.get_next_data_from_dataloader(dataloader, train)
                        batch_data.append(data)
                        batch_data_next.append(data_next)
                        data_idx = data_idx + 1


                    if train:
                        x_tensor_dict = self.process_data(batch_data)
                        x_tensor_dict_next = self.process_data(batch_data_next)
                    else:
                        x_tensor_dict = self.process_data_plan(batch_data)
                        x_tensor_dict_next = self.process_data_plan(batch_data_next)

                    if train:
                        batch_result_dict = self.training(
                                x_tensor_dict,
                                x_tensor_dict_next,
                                batch_size,
                                train=train,
                                threshold = threshold)
                    else:
                        batch_result_dict = self.planning(
                                x_tensor_dict,
                                x_tensor_dict_next,
                                batch_size,
                                train=train,
                                threshold = threshold)
                    

                if train:
                    self.save_checkpoint(e) 
        
            
        return batch_result_dict
