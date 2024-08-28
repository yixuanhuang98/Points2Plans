import numpy as np
import os
import pickle
import time
import copy
import math
import torch
import torch.nn as nn 

from itertools import permutations
from relational_dynamics.utils import math_util
from relational_dynamics.utils.other_util import LinearBlock, MLP, rotate_2d
from relational_dynamics.utils.data_utils import scale_min_max
from relational_dynamics.utils import torch_util


class PerSceneLoader(object):
    def __init__(self,  
                 scene_path, 
                 scene_type, 
                 relation_angle,
                 start = -1,
                 end = -1,
                 single_step_training = False, 
                 pick_place = False, 
                 push = False,
                 set_max = False,
                 train = True,
                 scene_pos_info=None,
                 max_objects = 5,
                 test_dir_1 = None,
                 POMDP_push = False,
                 add_noise_pc = False, 
                 sudo_pickplace = False, 
                 updated_behavior_params = False, 
                 use_shared_latent_embedding = False, 
                 use_seperate_latent_embedding = False,
                 online_planning = False,
                 push_3_steps = False,
                 train_object_identity = False,
                 use_boundary_relations = False,
                 consider_z_offset = False,
                 seperate_env_id = False,
                 max_env_num = 0,
                 env_first_step = False,
                 use_discrete_z = False,
                 fast_training = False,
                 one_bit_env = False,
                 rcpe = False,
                 pe = False,
                 bookshelf_env_shift = 0,
                 push_steps = 2, 
                 enable_return = False,
                 lfd_search = False,
                 get_hidden_label = False,
                 get_inside_relations = False,
                 enable_place_inside = False,
                 use_discrete_place = False, 
                 seperate_place = False, 
                 binary_grasp = False, 
                 open_close_drawer = False, 
                 softmax_identity = False, 
                 train_inside_feasibility = False, 
                 evaluate_pickplace = False,
                 this_one_hot_encoding = None,
                 ): 
        self.rcpe = rcpe
        self.pe = pe
        self.lfd_search = lfd_search

        
        self.single_step_training = single_step_training
        
        self.push_steps = push_steps
        
        self.evaluate_pickplace = evaluate_pickplace
        self.updated_behavior_params = updated_behavior_params
        self.train = train
        self.scene_path = scene_path
        self.scene_type = scene_type
        self.set_max = set_max
        self.pushing = push
        self.use_shared_latent_embedding = use_shared_latent_embedding
        self.use_seperate_latent_embedding = use_seperate_latent_embedding
        self.online_planning = online_planning
        self.push_3_steps = push_3_steps
        self.pick_place = pick_place
        
        self.train_object_identity = train_object_identity
        self.use_boundary_relations = use_boundary_relations
        self.consider_z_offset = consider_z_offset
        self.seperate_env_id = seperate_env_id
        self.max_env_num = max_env_num
        self.env_first_step = env_first_step
        self.use_discrete_z = use_discrete_z
        
        self.enable_return = enable_return
        self.POMDP_push = POMDP_push
        
        self.add_noise_pc = add_noise_pc
        
        self.sudo_pickplace = sudo_pickplace
        self.get_hidden_label = get_hidden_label

        self.fast_training = fast_training

        self.one_bit_env = one_bit_env

        self.bookshelf_env_shift = bookshelf_env_shift

        self.data_filter_leap = 1
        
        self.get_inside_relations = get_inside_relations
        self.enable_place_inside = enable_place_inside
        self.binary_grasp = binary_grasp
        self.open_close_drawer = open_close_drawer
        self.softmax_identity = softmax_identity
        self.train_inside_feasibility = train_inside_feasibility
        self.use_discrete_place = use_discrete_place
        self.seperate_place = seperate_place


        with open(self.scene_path, 'rb') as f:
            data, attrs = pickle.load(f)

        self.sample_time_step = []
        for each_step in range(data['point_cloud_1'].shape[0]):
            self.sample_time_step.append(each_step)
                
        self.params = {
            'theta_predicte_lr_fb_ab' : np.pi / relation_angle, # keep consistent for 45 degrees since there are some problems about 90 degrees. 
            'occ_IoU_threshold' : 0.5,
            'block_theta': np.pi * (15.0/32.0),  
        }


        self.all_point_cloud_last = []
        data_size = 128
        self.scale = 1
        
        self.all_pos_list_last = []
        self.all_orient_list = []
        self.all_orient_list_last = []
        self.all_point_cloud = []
        self.all_pos_list = []
        self.all_hidden_label_list = []
        self.all_pos_list_p = []
        self.all_gt_pose_list = []
        self.all_gt_orientation_list = []
        self.all_gt_max_pose_list = []
        self.all_gt_min_pose_list = []
        self.all_gt_extents_range_list = []
        self.all_gt_extents_list = []
        self.all_relation_list = []
        self.all_initial_bounding_box = []
        self.all_bounding_box = []
        self.all_axis_bounding_box = []
        self.all_rotated_bounding_box = []
        self.all_gt_identity_list = []
        self.all_rgb_identity_list = []
        self.all_gt_env_identity_list = []
        self.all_gt_grapable_list = []
        self.all_obj_boundary_list = []

        
        
        
        total_objects = 0
        if total_objects == 0:
            for k, v in data['objects'].items():
                if 'block' in k:
                    total_objects += 1

        
        
        
        if self.seperate_env_id:
            self.total_pure_obj_num = 0
            self.total_pure_env_num = 0
            for k, v in attrs['objects'].items():
                if 'block' in k and v['fix_base_link'] == False:
                    self.total_pure_obj_num += 1
                if 'block' in k and v['fix_base_link'] == True:
                    self.total_pure_env_num += 1
         



        self.total_objects = total_objects

        

        self.obj_pair_list = list(permutations(range(total_objects), 2))
        
        if self.single_step_training:
            self.start = start
            self.end = end
        else:
            self.start = 0
            self.end = data['point_cloud_1'].shape[0]
        for i in range(self.start, self.end):
            self.all_point_cloud.append([])
            self.all_pos_list.append([])
            self.all_gt_pose_list.append([])
            self.all_gt_identity_list.append([])
            self.all_gt_env_identity_list.append([])
            self.all_gt_grapable_list.append([])
            self.all_rgb_identity_list.append([])
            self.all_gt_orientation_list.append([])
            self.all_gt_max_pose_list.append([])
            self.all_gt_min_pose_list.append([])
            self.all_pos_list_p.append([])
            self.all_relation_list.append([])
            self.all_bounding_box.append([])
            self.all_axis_bounding_box.append([])
            self.all_rotated_bounding_box.append([])
            self.all_obj_boundary_list.append([])
        

        self.max_objects = max_objects
        A = np.arange(self.max_objects)
        
        if train:
            np.random.shuffle(A)

        select_obj_num_range = A[:total_objects]
        self.select_obj_num_range = select_obj_num_range
        one_hot_encoding = np.zeros((total_objects, self.max_objects))
        for i in range(len(select_obj_num_range)):
            one_hot_encoding[i][select_obj_num_range[i]] = 1
        if self.fast_training:
            self.one_hot_encoding_tensor_fast = torch.tensor(one_hot_encoding)
            self.total_objects_fast = total_objects
        
        
        block_string = 'block_'
        for j in range(total_objects):
            each_obj = j
            if total_objects >= 10 and each_obj + 1 < 10:
                current_block = "block_0" + str(each_obj + 1)
            else:
                current_block = "block_" + str(each_obj + 1)
            if 'extents' in attrs['objects'][current_block]: 
                if attrs['objects'][current_block]['extents_ranges'] == None:
                    attrs['objects'][current_block]['extents_ranges'] = [[attrs['objects'][current_block]['extents'][0], attrs['objects'][current_block]['extents'][0]], [attrs['objects'][current_block]['extents'][1], attrs['objects'][current_block]['extents'][1]], [attrs['objects'][current_block]['extents'][2], attrs['objects'][current_block]['extents'][2]]]
                self.all_gt_extents_range_list.append(attrs['objects'][current_block]['extents_ranges'])
                self.all_gt_extents_list.append(attrs['objects'][current_block]['extents'])
            else:
                self.all_gt_extents_list.append([attrs['objects'][current_block]['x_extent'], attrs['objects'][current_block]['y_extent'], attrs['objects'][current_block]['z_extent']])

        for i in range(self.start, self.end):
            if self.single_step_training:
                i_index = i - self.start
            else:
                i_index = i
            point_string = 'point_cloud_'
            block_string = 'block_'

            

            if True:
                if 'contact' in data:  
                    contact_array = np.zeros((total_objects, total_objects))

                    time_step = self.sample_time_step[i]
                    for contact_i in range(len(data['contact'][time_step])):
                        if self.bookshelf_env_shift > 0:
                            if data['contact'][time_step][contact_i]['body0'] > 0 and data['contact'][time_step][contact_i]['body0'] < total_objects + 1 and data['contact'][time_step][contact_i]['body1'] > 0 and data['contact'][time_step][contact_i]['body1'] < total_objects + 1:
                                contact_array[data['contact'][time_step][contact_i]['body0'] - 1, data['contact'][time_step][contact_i]['body1'] - 1] = 1
                                contact_array[data['contact'][time_step][contact_i]['body1'] - 1, data['contact'][time_step][contact_i]['body0'] - 1] = 1
                        else:
                            if data['contact'][time_step][contact_i]['body0'] > -1 and data['contact'][time_step][contact_i]['body0'] < total_objects and data['contact'][time_step][contact_i]['body1'] > -1 and data['contact'][time_step][contact_i]['body1'] < total_objects:
                                contact_array[data['contact'][time_step][contact_i]['body0'], data['contact'][time_step][contact_i]['body1']] = 1
                                contact_array[data['contact'][time_step][contact_i]['body1'], data['contact'][time_step][contact_i]['body0']] = 1

                for j in range(total_objects):
                    each_obj = j
                    if total_objects >= 10 and each_obj + 1 < 10:
                        current_block = "block_0" + str(each_obj + 1)
                    else:
                        current_block = "block_" + str(each_obj + 1)
                    initial_bounding_box = []
                    TF_matrix = []
                    for inner_i in range(2):
                        for inner_j in range(2):
                            for inner_k in range(2):
                                if True:
                                    step = 0
                                    if 'extents' in attrs['objects'][current_block]: #self.pushing and not self.pick_place: self.pushing:
                                        initial_bounding_box.append(math_util.pose_to_homogeneous(np.array([data['objects'][current_block]['position'][step][0] + ((inner_i*2) - 1)*attrs['objects'][current_block]['extents'][0]/2, 
                                        data['objects'][current_block]['position'][step][1] + ((inner_j*2) - 1)*attrs['objects'][current_block]['extents'][1]/2, 
                                        data['objects'][current_block]['position'][step][2] + ((inner_k*2) - 1)*attrs['objects'][current_block]['extents'][2]/2]), np.array([0,0,0,1])))
                                        TF_matrix.append(np.matmul(math_util.homogeneous_transpose(math_util.pose_to_homogeneous(data['objects'][current_block]['position'][0], np.array([0,0,0,1]))), initial_bounding_box[-1]))
                                    else:
                                        
                                        initial_bounding_box.append(math_util.pose_to_homogeneous(np.array([data['objects'][current_block]['position'][step][0] + ((inner_i*2) - 1)*attrs['objects'][current_block]['x_extent']/2, 
                                        data['objects'][current_block]['position'][step][1] + ((inner_j*2) - 1)*attrs['objects'][current_block]['y_extent']/2, 
                                        data['objects'][current_block]['position'][step][2] + ((inner_k*2) - 1)*attrs['objects'][current_block]['z_extent']/2]), np.array([0,0,0,1])))
                                        TF_matrix.append(np.matmul(math_util.homogeneous_transpose(math_util.pose_to_homogeneous(data['objects'][current_block]['position'][0], np.array([0,0,0,1]))), initial_bounding_box[-1]))
                                    
                    
                    initial_bounding_box = np.array(initial_bounding_box)
                    
                    rotated_bounding_box = np.zeros((initial_bounding_box.shape[0], initial_bounding_box.shape[1], initial_bounding_box.shape[2]))
                    TF_rotated_bounding_matrix = []
                    
                    for inner_i in range(initial_bounding_box.shape[0]):
                        rotated_bounding_box[inner_i, :, :] = math_util.pose_to_homogeneous(data['objects'][current_block]['position'][0], data['objects'][current_block]['orientation'][0])@TF_matrix[inner_i]
                        TF_rotated_bounding_matrix.append(np.matmul(math_util.homogeneous_transpose(math_util.pose_to_homogeneous(data['objects'][current_block]['position'][0], data['objects'][current_block]['orientation'][0])), rotated_bounding_box[inner_i, :, :]))
                    
                    self.all_rotated_bounding_box[i_index].append(np.array(TF_rotated_bounding_matrix))

                    final_bounding_box = np.zeros((initial_bounding_box.shape[0], initial_bounding_box.shape[1], initial_bounding_box.shape[2]))
                    final_array = np.zeros((initial_bounding_box.shape[0], 3))
                    if True:
                        for inner_i in range(rotated_bounding_box.shape[0]): 
                            final_bounding_box[inner_i,:, :] = math_util.pose_to_homogeneous(data['objects'][current_block]['position'][self.sample_time_step[i]], data['objects'][current_block]['orientation'][self.sample_time_step[i]])@TF_rotated_bounding_matrix[inner_i]
                            final_array[inner_i, :] = math_util.homogeneous_to_position(final_bounding_box[inner_i, :, :])
                    
                        
                      

                    self.all_bounding_box[i_index].append(final_array)
                    
                    max_current_pose = np.max(final_array, axis = 0)[:3]
                    min_current_pose = np.min(final_array, axis = 0)[:3]

                    max_min_extents = max_current_pose - min_current_pose
                    axis_final_array = [] #np.zeros((final_array.shape[0], final_array.shape[1]))

                    for max_min_i in range(2):
                        for max_min_j in range(2):
                            for max_min_z in range(2):
                                axis_final_array.append([min_current_pose[0] + max_min_i * max_min_extents[0], min_current_pose[1] + max_min_j * max_min_extents[1], min_current_pose[2] + max_min_z * max_min_extents[2]])
                    axis_final_array = np.array(axis_final_array)
                    
                    self.all_axis_bounding_box[i_index].append(axis_final_array)
                    
                    
                    
                    self.all_gt_max_pose_list[i_index].append(max_current_pose)
                    self.all_gt_min_pose_list[i_index].append(min_current_pose)

                    self.all_obj_boundary_list[i_index].append(max_current_pose - min_current_pose)

                    if self.add_noise_pc:
                        self.all_point_cloud[i_index].append(data[point_string + str(j+1) + 'sampling_noise'][i][:data_size, :])
                    else:
                        self.all_point_cloud[i_index].append(data[point_string + str(j+1) + 'sampling'][i][:data_size, :])
            
                    
                    self.all_pos_list[i_index].append(self.get_point_cloud_center(data[point_string + str(j+1) + 'sampling'][i])) ## consider the case with memory
                        
                    
                    if True:
                        if True:
                            identity_max_objects = 10
                            rgb_identity_max = 10
                            env_identity_max = 2
                            current_identity_list = []
                            rgb_identity_list = []
                            if each_obj >= 3:
                                current_identity_list = [1,0,0,0,0,0,0,0,0,0]
                            else:
                                for object_identity_id in range(identity_max_objects):
                                    if object_identity_id == each_obj:
                                        current_identity_list.append(1)
                                    else:
                                        current_identity_list.append(0)
                                
                            
                            for rgb_identity_id in range(rgb_identity_max):
                                if rgb_identity_id == each_obj:
                                    rgb_identity_list.append(1)
                                else:
                                    rgb_identity_list.append(0)

                            if self.one_bit_env:
                                if self.seperate_env_id:
                                    if self.select_obj_num_range[each_obj] < (max_objects - self.max_env_num):
                                        env_identity_list = [0]
                                    else:
                                        env_identity_list = [1]
                                else:
                                    if self.open_close_drawer:
                                        
                                        if attrs['objects'][current_block]['object_type'] == 'urdf' and 'drawer' in attrs['objects'][current_block]['asset_filename']:
                                            ## yixuan note here, always assume drawer is closed
                                            if np.abs(data['objects'][current_block]['position'][self.sample_time_step[i]][0] - data['objects'][current_block]['position'][0][0]) > 0.02:
                                                env_identity_list = [0, 1, 0]
                                            else:
                                                # env_identity_list = [0, 1, 1]
                                                env_identity_list = [0, 0, 1]
                                        elif attrs['objects'][current_block]['fix_base_link']:
                                            env_identity_list = [0, 0, 0]
                                        else:
                                            env_identity_list = [1, 0, 0]
                                    else:
                                        # if self.get_hidden_label and data['hidden_label'][i][each_obj] == 1:
                                        #     env_identity_list = [0] ## this does not work since it's hard to determine based on a partial view point cloud
                                        if attrs['objects'][current_block]['fix_base_link']:
                                            env_identity_list = [0]
                                        else:
                                            env_identity_list = [1]
                            else:
                                if self.seperate_env_id:
                                    if self.select_obj_num_range[each_obj] < (max_objects - self.max_env_num):
                                        env_identity_list = [0,1]
                                    else:
                                        env_identity_list = [1,0]
                                else:
                                    if attrs['objects']['block_'+str(each_obj+1)]['fix_base_link']:
                                        env_identity_list = [0,1]
                                    else:
                                        env_identity_list = [1,0]

                            
                            self.all_rgb_identity_list[i_index].append(rgb_identity_list)
                            self.all_gt_identity_list[i_index].append(current_identity_list)
                            self.all_gt_env_identity_list[i_index].append(env_identity_list)
                            self.all_gt_pose_list[i_index].append(data['objects'][current_block]['position'][self.sample_time_step[i]])
                            self.all_gt_orientation_list[i_index].append(data['objects'][current_block]['orientation'][self.sample_time_step[i]])
                        

                if self.get_hidden_label:
                    self.all_hidden_label_list.append(data['hidden_label'][i])
                
                self.use_shelf = False
                if self.all_gt_env_identity_list[0][-1][0] == 0 and self.all_gt_env_identity_list[0][-2][0] == 0:
                    if self.all_gt_extents_list[-1][-1] < 0.06 and self.all_gt_extents_list[-2][-1] < 0.06:
                        self.use_shelf = True
                
                for obj_pair in self.obj_pair_list:
                    (anchor_idx, other_idx) = obj_pair
                    if True:
                        self.all_relation_list[i_index].append(self.get_predicates(contact_array, anchor_idx, other_idx ,self.all_axis_bounding_box[i_index][anchor_idx], self.all_gt_pose_list[i_index][anchor_idx], self.all_gt_max_pose_list[i_index][anchor_idx], self.all_gt_min_pose_list[i_index][anchor_idx], self.all_gt_extents_list[anchor_idx], self.all_axis_bounding_box[i_index][other_idx], self.all_gt_pose_list[i_index][other_idx], self.all_gt_max_pose_list[i_index][other_idx],self.all_gt_min_pose_list[i_index][other_idx][:], self.all_gt_extents_list[other_idx])[:])
                        if self.binary_grasp:
                            self.all_gt_grapable_list[i_index].append(self.get_constrained_predicates(contact_array, anchor_idx, other_idx ,self.all_axis_bounding_box[i_index][anchor_idx], self.all_gt_pose_list[i_index][anchor_idx], self.all_gt_max_pose_list[i_index][anchor_idx], self.all_gt_min_pose_list[i_index][anchor_idx], self.all_gt_extents_list[anchor_idx], self.all_axis_bounding_box[i_index][other_idx], self.all_gt_pose_list[i_index][other_idx], self.all_gt_max_pose_list[i_index][other_idx],self.all_gt_min_pose_list[i_index][other_idx][:], self.all_gt_extents_list[other_idx])[:])
                
                min_value = 100
                for each_pose_id in range(len(self.all_gt_pose_list[i_index]) - 2):
                    if self.all_gt_pose_list[i_index][each_pose_id][2] >= 0.3:
                        if self.all_gt_pose_list[i_index][each_pose_id][0] < min_value:
                            min_value = self.all_gt_pose_list[i_index][each_pose_id][0]
                
                for j in range(total_objects):
                    
                    each_obj = j
                    if total_objects >= 10 and each_obj + 1 < 10:
                        current_block = "block_0" + str(each_obj + 1)
                    else:
                        current_block = "block_" + str(each_obj + 1)
                    
                    graspable_identity = [0]
            
        
        self.action_1 = []

        
        if self.use_boundary_relations or self.POMDP_push or self.sudo_pickplace:
            if self.push_3_steps:
                if self.sudo_pickplace:
                    
                    self.all_action_list = []
                    self.move_obj_list = [] 
                    for each_action_step in range(self.start, self.end - 1):
                        if self.single_step_training:
                            each_action_step_id = each_action_step - self.start
                        else:
                            each_action_step_id = each_action_step
                        self.all_action_list.append([])
                        current_action_list = attrs['sudo_action_list'][each_action_step]
                        
                        if current_action_list[0] == 'pickplace' or current_action_list[0] == 'stack':
                            for i in range(self.max_objects):
                                self.all_action_list[each_action_step_id].append(0)
                            for i in range(total_objects):
                                if str(i+1) in current_action_list[1]:
                                    move_obj = i
                            self.move_obj_list.append(move_obj)

                            if self.train:
                                self.all_action_list[each_action_step_id][select_obj_num_range[move_obj]] = 1
                            else:
                                self.all_action_list[each_action_step_id][move_obj] = 1
                            for coninuous_i in range(len(current_action_list[2])):
                                self.all_action_list[each_action_step_id].append(current_action_list[2][coninuous_i])
                            self.all_action_list[each_action_step_id].insert(0, 0)
                            assert self.all_action_list[each_action_step_id][0] == 0
                        elif current_action_list[0] == 'push' or current_action_list[0] == 'pull':
                            for i in range(self.max_objects):
                                self.all_action_list[each_action_step_id].append(0)
                            for i in range(total_objects):
                                if str(i+1) in current_action_list[1]:
                                    move_obj = i
                            self.move_obj_list.append(move_obj)
                            if self.train:
                                self.all_action_list[each_action_step_id][select_obj_num_range[move_obj]] = 1
                            else:
                                self.all_action_list[each_action_step_id][move_obj] = 1
                            for coninuous_i in range(len(current_action_list[2])):
                                self.all_action_list[each_action_step_id].append(current_action_list[2][coninuous_i])
                            self.all_action_list[each_action_step_id].insert(0, 1)
                    
                    
        
        if self.use_discrete_place:
            
            relation_arr_list = []
            for each_step_id in range(len(self.all_relation_list)):
                relation_arr_list.append(np.array(self.all_relation_list[each_step_id]))

            self.buffer_tensor_0 = torch.zeros(len(self.all_relation_list) - 1, total_objects, 128)
            self.support_surface_id = []
            for each_step_id in range(1, len(self.all_relation_list)):
                
                self.support_surface_id.append([])
                
                if self.all_gt_pose_list[each_step_id][self.move_obj_list[0]][-1] - self.all_gt_pose_list[each_step_id - 1][self.move_obj_list[0]][-1] > 0.10:
                    
                    if not np.array_equal(relation_arr_list[each_step_id][:, 6], relation_arr_list[each_step_id - 1][:, 6]):
                        diff = relation_arr_list[each_step_id][:, 6] - relation_arr_list[each_step_id - 1][:, 6]
                        if np.max(diff) > 0:
                            index = np.where(diff == 1.0)
                            if len(index[0]) == 2:
                                for index_value in index[0]:
                                    current_obj_id = (int)(index_value/(total_objects - 1))
                                    other_obj_id = index_value % (total_objects - 1)
                                    if other_obj_id >= current_obj_id:
                                        other_obj_id = other_obj_id + 1
                                    if current_obj_id == self.move_obj_list[0] and self.all_gt_env_identity_list[0][other_obj_id][0] == 0:
                                        for each_obj_i in range(total_objects):
                                            self.support_surface_id[each_step_id - 1].append(other_obj_id)
                            else:
                                self.data_filter_leap = 0
        


        
        self.obj_voxels_single = [] 
        for i in range(self.start, self.end):
            if self.fast_training:
                self.obj_voxels_single.append([])
            else:
                self.obj_voxels_single.append(dict())
        
        
        for obj_id in range(self.total_objects):
            for i in range(self.start, self.end):
                if self.single_step_training:
                    total_point_cloud = self.all_point_cloud[i - self.start][obj_id]
                else:
                    total_point_cloud = self.all_point_cloud[i][obj_id]
            
                if self.fast_training:
                    if self.single_step_training:
                        self.obj_voxels_single[i - self.start].append(total_point_cloud.T)
                    else:
                        self.obj_voxels_single[i].append(total_point_cloud.T)
                else:
                    self.obj_voxels_single[i][obj_id] = total_point_cloud.T 
        
        if train:
            self.maximum_multi_step = 1
            self.relation_list_effective_part = []
            relation_index_list = 0

            self.obj_voxels_single = torch.Tensor(np.array(self.obj_voxels_single).astype(np.float64))
            self.all_relation_list = torch.Tensor(self.all_relation_list)
            
            for i in range(self.max_objects):
                for j in range(self.max_objects):
                    if j != i:
                        if i < self.obj_voxels_single.shape[1] and j < self.obj_voxels_single.shape[1]:
                            self.relation_list_effective_part.append(relation_index_list)
                        relation_index_list += 1
            
            relation_list_final = -torch.ones(1, self.maximum_multi_step + 1, self.max_objects * (self.max_objects - 1), self.all_relation_list.shape[2])
            relation_list_final[:, :self.all_relation_list.shape[0], self.relation_list_effective_part, :] = self.all_relation_list
            
            self.src_key_padding_mask = []
            for i in range(self.max_objects):
                if i < self.obj_voxels_single.shape[1]:
                    self.src_key_padding_mask.append(0)
                else:
                    self.src_key_padding_mask.append(1)
            
            self.all_relation_fast = relation_list_final

            self.obj_voxels_single_fast = torch.rand(1, self.maximum_multi_step + 1, self.max_objects,self.obj_voxels_single.shape[2], self.obj_voxels_single.shape[3])

            # self.obj_voxels_single_fast = torch.Tensor(self.obj_voxels_single)
            self.obj_voxels_single_fast[0, :self.obj_voxels_single.shape[0], :self.obj_voxels_single.shape[1], :, :] = torch.Tensor(self.obj_voxels_single)
            
            
            self.all_gt_grapable_list = torch.Tensor(self.all_gt_grapable_list)
            gt_grapable_list_final = -torch.ones(1, self.maximum_multi_step + 1, self.max_objects * (self.max_objects - 1), self.all_gt_grapable_list.shape[2])
            gt_grapable_list_final[:, :self.all_gt_grapable_list.shape[0], self.relation_list_effective_part, :] = self.all_gt_grapable_list
            
            self.all_gt_grapable_list_fast = gt_grapable_list_final


            # self.env_identity_list_fast = torch.Tensor(self.all_gt_env_identity_list)

            self.env_identity_torch = torch.Tensor(self.all_gt_env_identity_list)
            self.env_identity_list_fast = -torch.ones(1, self.maximum_multi_step + 1, self.max_objects, self.env_identity_torch.shape[2])
            self.env_identity_list_fast[:, :self.env_identity_torch.shape[0], :self.env_identity_torch.shape[1], :] = self.env_identity_torch

            
            ### do not change 
            self.all_obj_boundary_list_fast = torch.Tensor(np.array(self.all_obj_boundary_list))
            self.all_hidden_label_fast = torch.Tensor(self.all_hidden_label_list).type(torch.int)
            new_latent = np.zeros((4, 1, 3, 256))
            self.new_latent = torch.Tensor(new_latent)
            ### 

            if self.use_discrete_place:
                self.support_surface_id_fast = []
                assert len(self.support_surface_id) == 1
                for each_id in range(len(self.support_surface_id)):
                    self.support_surface_id_fast.append(torch.Tensor([self.support_surface_id[each_id]]).type(torch.int))

            else:
                self.support_surface_id_fast = -torch.ones(total_objects, 128)     
                self.buffer_tensor_0 = -torch.ones(total_objects, 128)    

            
            
            if not self.single_step_training:
                assert data['point_cloud_1'].shape[0] == self.obj_voxels_single_fast.shape[1]
            else:
                assert self.obj_voxels_single_fast.shape[1] == 2
            
            
            nodes = list(range(self.max_objects))
            # Create a completely connected graph
            edges = list(permutations(nodes, 2))
            edge_index = torch.LongTensor(np.array(edges).T)
            self.edge_attr_fast = edge_index
            
            self.one_hot_encoding_tensor_fast_batch = -torch.ones(1, self.max_objects, self.one_hot_encoding_tensor_fast.shape[1])
            self.one_hot_encoding_tensor_fast_batch[:, :self.one_hot_encoding_tensor_fast.shape[0], :] = self.one_hot_encoding_tensor_fast
            
            
            
            self.all_action_label_fast = -np.ones((1, self.maximum_multi_step))
            for i in range(len(self.all_action_list)):
                self.all_action_label_fast[0, i] = self.all_action_list[i][0]

            self.all_action_fast_temp = torch.Tensor(self.all_action_list)
            self.all_action_fast = -torch.ones((1, self.maximum_multi_step, self.max_objects, self.all_action_fast_temp.shape[1]))
            for i in range(self.all_action_fast_temp.shape[0]):
                for j in range(self.max_objects):
                    self.all_action_fast[:, i, j, :] = self.all_action_fast_temp[i]

           
            if self.rcpe or self.pe:
                current_quaternian = torch.Tensor(np.array(self.all_gt_orientation_list))
                
                self.current_quaternian = current_quaternian

                gt_pose_list = torch.Tensor(np.array(self.all_gt_pose_list))
                self.current_position = gt_pose_list
                self.all_gt_extents_list_tensor = torch.Tensor(self.all_gt_extents_list)
                
                

                for each_action_id in range(len(self.all_action_list)):
                    move_obj_id = self.move_obj_list[each_action_id]
                    test_pose_current = self.all_gt_pose_list[each_action_id][move_obj_id]
                    test_pose_next = self.all_gt_pose_list[each_action_id + 1][move_obj_id]
                    test_pose_difference = test_pose_next - test_pose_current
                    
                    if np.linalg.norm(test_pose_difference[:2] - np.array(self.all_action_list[each_action_id][-3: -1])) > 0.03:
                        self.data_filter_leap = 0

                current_position = gt_pose_list.detach().numpy()
            
                current_pose = torch.Tensor(current_position)

                
                self.current_pose = current_pose
                self.current_pose_fast = -torch.ones(1, self.maximum_multi_step + 1, self.max_objects, self.current_pose.shape[2])
                self.current_pose_fast[:, :self.current_pose.shape[0], :self.current_pose.shape[1], :] = self.current_pose

            
        else:
            self.all_relation_fast = torch.Tensor(self.all_relation_list)
            self.obj_voxels_single_fast = torch.Tensor(np.array(self.obj_voxels_single).astype(np.float64))
            self.env_identity_list_fast = torch.Tensor(self.all_gt_env_identity_list)
            self.all_obj_boundary_list_fast = torch.Tensor(np.array(self.all_obj_boundary_list))
            self.all_hidden_label_fast = torch.Tensor(self.all_hidden_label_list).type(torch.int)
            if self.use_discrete_place:
                self.support_surface_id_fast = []
                for each_id in range(len(self.support_surface_id)):
                    self.support_surface_id_fast.append(torch.Tensor([self.support_surface_id[each_id]]).type(torch.int))

            else:
                self.support_surface_id_fast = -torch.ones(total_objects, 128)     
                self.buffer_tensor_0 = -torch.ones(total_objects, 128)    

            self.all_gt_grapable_list_fast = torch.Tensor(self.all_gt_grapable_list)
            
            if not self.single_step_training:
                assert data['point_cloud_1'].shape[0] == self.obj_voxels_single_fast.shape[0]
            else:
                assert self.obj_voxels_single_fast.shape[0] == 2
            assert total_objects == self.obj_voxels_single_fast.shape[1]
            new_latent = np.zeros((4, 1, 3, 256))
            self.new_latent = torch.Tensor(new_latent)
            nodes = list(range(self.total_objects_fast))
            # Create a completely connected graph
            edges = list(permutations(nodes, 2))
            edge_index = torch.LongTensor(np.array(edges).T)
            self.edge_attr_fast = edge_index
            
            self.all_action_label_fast = []
            for i in range(len(self.all_action_list)):
                self.all_action_label_fast.append((int)(self.all_action_list[i][0]))
            
            self.all_action_fast_temp = torch.Tensor(self.all_action_list)
            self.all_action_fast = torch.zeros((self.all_action_fast_temp.shape[0], self.total_objects_fast, self.all_action_fast_temp.shape[1]))
            for i in range(self.total_objects_fast):
                self.all_action_fast[:, i, :] = self.all_action_fast_temp

            self.one_hot_encoding_tensor_fast_batch = self.one_hot_encoding_tensor_fast
            
            if self.rcpe or self.pe:
                current_quaternian = torch.Tensor(np.array(self.all_gt_orientation_list))
                self.current_quaternian = current_quaternian

                gt_pose_list = torch.Tensor(np.array(self.all_gt_pose_list))
                self.current_position = gt_pose_list
                self.all_gt_extents_list_tensor = torch.Tensor(self.all_gt_extents_list)

                for each_action_id in range(len(self.all_action_list)):
                    move_obj_id = self.move_obj_list[each_action_id]
                    test_pose_current = self.all_gt_pose_list[each_action_id][move_obj_id]
                    test_pose_next = self.all_gt_pose_list[each_action_id + 1][move_obj_id]
                    test_pose_difference = test_pose_next - test_pose_current
                    if np.linalg.norm(test_pose_difference[:2] - np.array(self.all_action_list[each_action_id][-3: -1])) > 0.03:
                        self.data_filter_leap = 0
                    
                
                
                current_position = gt_pose_list.detach().numpy()
            
                
                current_pose = torch.Tensor(current_position)

                self.current_pose_fast = current_pose
                
        
    def get_predicates(self, contact_arr, anchor_id, other_id, anchor_bounding_box ,anchor_pose, anchor_pose_max, anchor_pose_min, anchor_extents, other_bounding_box, other_pose, other_pose_max, other_pose_min, other_extents): # to start, assume no orientation
        action = []

        cf_o1_bbox_corners = np.concatenate([anchor_bounding_box, np.expand_dims(anchor_pose, axis=0)], axis=0).T

        cf_o2_bbox_corners = np.concatenate([other_bounding_box, np.expand_dims(other_pose, axis=0)], axis=0).T

        
        o1_predicate_args = {
            'obj_id' : anchor_id,
            'cf_bbox_corners' : cf_o1_bbox_corners, 
        }
        o2_predicate_args = {
            'obj_id' : other_id,
            'cf_bbox_corners' : cf_o2_bbox_corners,
        }

        self.compute_left_right_predicates(o1_predicate_args, o2_predicate_args, action)

        
        self.compute_above_below_predicates(o1_predicate_args, o2_predicate_args, action)

        self.compute_front_behind_predicates(o1_predicate_args, o2_predicate_args, action)



        if((other_pose[2] - anchor_pose[2]) > 0):
            current_extents = np.array(anchor_pose_max) - np.array(anchor_pose_min)
        else:
            current_extents = np.array(other_pose_max) - np.array(other_pose_min)
        
        

        sudo_contact = 0
        if np.abs(other_pose[2] - anchor_pose[2]) > 0.04 and np.abs(other_pose[2] - anchor_pose[2]) < 0.12:
            if np.abs(other_pose[0] - anchor_pose[0]) < current_extents[0]/2 and np.abs(other_pose[1] - anchor_pose[1]) < current_extents[1]/2:
                sudo_contact = 1
          
        if contact_arr[0][0] == -1: # simple trick to deal with unsaved contact relations
            action.append(sudo_contact)
        else:
            action.append(contact_arr[anchor_id][other_id])

        
        if self.use_boundary_relations:
            
            
            pair_corner = [[[other_bounding_box[0][0], other_bounding_box[0][1]], [other_bounding_box[3][0], other_bounding_box[3][1]]], 
            [[other_bounding_box[0][0], other_bounding_box[0][1]], [other_bounding_box[5][0], other_bounding_box[5][1]]], 
            [[other_bounding_box[7][0], other_bounding_box[7][1]], [other_bounding_box[3][0], other_bounding_box[3][1]]], 
            [[other_bounding_box[7][0], other_bounding_box[7][1]], [other_bounding_box[5][0], other_bounding_box[5][1]]]]
            self.boundary_length = 0.10
            all_dist_list = []
            if action[5] == 1 and current_extents[0] > 0.2 and current_extents[1] > 0.2:
                for each_pair_corner in pair_corner:
                    all_dist_list.append(self.get_distance_from_point_to_line(anchor_pose[:2], each_pair_corner[0], each_pair_corner[1]))
                if min(all_dist_list) < self.boundary_length:
                    action.append(1)
                else:
                    action.append(0)
            else:
                action.append(0)

            # if action[5] == 1:
            #     if np.abs(anchor_pose[0] - other_pose[0]) > (current_extents[0]/2 - self.boundary_length) or np.abs(anchor_pose[1] - other_pose[1]) > (current_extents[1]/2 - self.boundary_length):
            #         action.append(1)
            #     else:
            #         action.append(0)
            # else:
            #     action.append(0)

        if self.get_inside_relations:
            inside_leap = 1
            for each_dim_id in range(3):
                buffer_z = 0.03
                if each_dim_id == 2:
                    if not (np.min(anchor_bounding_box[:, each_dim_id]) < (np.max(other_bounding_box[:, each_dim_id]) - buffer_z) and (np.min(anchor_bounding_box[:, each_dim_id]) + buffer_z) > np.min(other_bounding_box[:, each_dim_id]) and np.max(anchor_bounding_box[:, each_dim_id]) < (np.max(other_bounding_box[:, each_dim_id]) + 0.14)):
                        inside_leap = 0
                else:
                    if not (np.max(anchor_bounding_box[:, each_dim_id]) < np.max(other_bounding_box[:, each_dim_id]) and np.min(anchor_bounding_box[:, each_dim_id]) > np.min(other_bounding_box[:, each_dim_id])):
                        inside_leap = 0
            

            action.append(inside_leap)      
            
        
        
        return action

    def get_constrained_predicates(self, contact_arr, anchor_id, other_id, anchor_bounding_box ,anchor_pose, anchor_pose_max, anchor_pose_min, anchor_extents, other_bounding_box, other_pose, other_pose_max, other_pose_min, other_extents): # to start, assume no orientation
        action = []
        if self.use_shelf and self.binary_grasp and self.train_inside_feasibility:
            if anchor_pose[2] < 0.3 or other_pose[2] < 0.3:
                action.append(0) 
            elif self.all_gt_env_identity_list[0][anchor_id][0] == 0 or self.all_gt_env_identity_list[0][other_id][0] == 0:
                action.append(0) 
            else:
                inside_leap = 1
                for each_dim_id in range(3):
                    buffer_z = 0.03
                    if each_dim_id == 2:
                        if not (np.min(anchor_bounding_box[:, each_dim_id]) < (np.max(other_bounding_box[:, each_dim_id]) - buffer_z) and (np.min(anchor_bounding_box[:, each_dim_id]) + buffer_z) > np.min(other_bounding_box[:, each_dim_id]) and np.max(anchor_bounding_box[:, each_dim_id]) < (np.max(other_bounding_box[:, each_dim_id]) + 0.14)):
                            inside_leap = 0
                    else:
                        if not (np.max(anchor_bounding_box[:, each_dim_id]) < np.max(other_bounding_box[:, each_dim_id]) and np.min(anchor_bounding_box[:, each_dim_id]) > np.min(other_bounding_box[:, each_dim_id])):
                            inside_leap = 0
                action.append(inside_leap)    
        else:
            action.append(0)   
        

        if action[-1] == 1:
            action.append(0)   
        elif self.use_shelf and self.binary_grasp:
            if anchor_pose[2] < 0.3 or other_pose[2] < 0.3:
                action.append(0) 
            elif self.all_gt_env_identity_list[0][anchor_id][0] == 0 or self.all_gt_env_identity_list[0][other_id][0] == 0:
                action.append(0) 
            else:
                cf_o1_bbox_corners = np.concatenate([anchor_bounding_box, np.expand_dims(anchor_pose, axis=0)], axis=0).T

                cf_o2_bbox_corners = np.concatenate([other_bounding_box, np.expand_dims(other_pose, axis=0)], axis=0).T

                o1_predicate_args = {
                    'obj_id' : anchor_id,
                    'cf_bbox_corners' : cf_o1_bbox_corners, 
                }
                o2_predicate_args = {
                    'obj_id' : other_id,
                    'cf_bbox_corners' : cf_o2_bbox_corners,
                }

                self.compute_right_only_predicates(o1_predicate_args, o2_predicate_args, action)
        else:
            action.append(0)  

        return action

    

    def get_distance_from_point_to_line(self, point, line_point1, line_point2):
        if line_point1 == line_point2:
            point_array = np.array(point )
            point1_array = np.array(line_point1)
            return np.linalg.norm(point_array -point1_array )
        A = line_point2[1] - line_point1[1]
        B = line_point1[0] - line_point2[0]
        C = (line_point1[1] - line_point2[1]) * line_point1[0] + \
            (line_point2[0] - line_point1[0]) * line_point1[1]
        distance = np.abs(A * point[0] + B * point[1] + C) / (np.sqrt(A**2 + B**2))
        return distance

    
    def compute_left_right_predicates(self, o1_predicate_args, o2_predicate_args, predicates):
        """ Compute left-right predicates.
            Use camera frame coordinates.
            Relation rules:
                1) o1 center MUST be in half-space defined by o2 UPPER corner and theta (xz plane)
                2) o1 center MUST be in half-space defined by o2 LOWER corner and theta (xz plane)
                3) do same as 1) for xy
                4) do same as 2) for xy
                5) o1 center MUST be to left of all o2 corners
                6) All o1 corners MUST be to the left of o2 center
        """

        def left_of(cf_o1_bbox_corners, cf_o2_bbox_corners):

            # Check xz
            o1_xz_center = cf_o1_bbox_corners[[0,2], 8] # [x,z]

            # Get camera-frame axis-aligned bbox corners for o2. Only for x-z plane, and two left-most corners. 
            o2_upper_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[2,:8].max()]) # [x,z]
            o2_lower_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[2,:8].min()]) # [x,z]

            # Upper half-space defined by p'n + d = 0
            upper_normal = rotate_2d(np.array([-1,0]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            upper_d = -1 * o2_upper_corner.dot(upper_normal)
            first_rule = o1_xz_center.dot(upper_normal) + upper_d >= 0

            # Lower half-space defined by p'n + d = 0
            lower_normal = rotate_2d(np.array([0,1]), self.params['theta_predicte_lr_fb_ab'])
            lower_d = -1 * o2_lower_corner.dot(lower_normal)
            second_rule = o1_xz_center.dot(lower_normal) + lower_d >= 0

            xz_works = first_rule and second_rule

            # Check xy
            o1_xy_center = cf_o1_bbox_corners[[0,1], 8] # [x,y]

            # Get camera-frame axis-aligned bbox corners for o2. Only for x-y plane, and two left-most corners. 
            o2_upper_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[1,:8].max()]) # [x,y]
            o2_lower_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[1,:8].min()]) # [x,y]

            # Upper half-space defined by p'n + d = 0
            upper_normal = rotate_2d(np.array([-1,0]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            upper_d = -1 * o2_upper_corner.dot(upper_normal)
            third_rule = o1_xy_center.dot(upper_normal) + upper_d >= 0

            # Lower half-space defined by p'n + d = 0
            lower_normal = rotate_2d(np.array([0,1]), self.params['theta_predicte_lr_fb_ab'])
            lower_d = -1 * o2_lower_corner.dot(lower_normal)
            fourth_rule = o1_xy_center.dot(lower_normal) + lower_d >= 0

            xy_works = third_rule and fourth_rule

            # All corners check
            fifth_rule = np.all(o1_xz_center[0] <= cf_o2_bbox_corners[0,:8].min())

            # o1 right corners check
            sixth_rule = np.all(cf_o1_bbox_corners[0, :8].max() <= cf_o2_bbox_corners[0,8])

            return xz_works and xy_works and fifth_rule and sixth_rule

        obj1_id = o1_predicate_args['obj_id']
        obj2_id = o2_predicate_args['obj_id']

        # For symmetry, check if o1 is left of o2, and if o2 is right of o1
        cf_o1_bbox_corners = o1_predicate_args['cf_bbox_corners'].copy()
        cf_o2_bbox_corners = o2_predicate_args['cf_bbox_corners'].copy()
        o1_left_of_o2 = left_of(cf_o1_bbox_corners, cf_o2_bbox_corners)

        o2_left_of_o1 = left_of(cf_o2_bbox_corners, cf_o1_bbox_corners)

        cf_o1_bbox_corners[0,:] = cf_o1_bbox_corners[0,:] * -1
        cf_o2_bbox_corners[0,:] = cf_o2_bbox_corners[0,:] * -1
        o2_right_of_o1 = left_of(cf_o2_bbox_corners, cf_o1_bbox_corners)

        o1_right_of_o2 = left_of(cf_o1_bbox_corners, cf_o2_bbox_corners)

        if o1_left_of_o2 or o2_right_of_o1:
            predicates.append(1)
            predicates.append(0)  
        elif o2_left_of_o1 or o1_right_of_o2:
            predicates.append(0)
            predicates.append(1)  
        else:
            predicates.append(0)
            predicates.append(0)                


    def compute_block_only_predicates(self, o1_predicate_args, o2_predicate_args, predicates):
        """ Compute left-right predicates.
            Use camera frame coordinates.
            Relation rules:
                1) o1 center MUST be in half-space defined by o2 UPPER corner and theta (xz plane)
                2) o1 center MUST be in half-space defined by o2 LOWER corner and theta (xz plane)
                3) do same as 1) for xy
                4) do same as 2) for xy
                5) o1 center MUST be to left of all o2 corners
                6) All o1 corners MUST be to the left of o2 center
        """

        def left_of(cf_o1_bbox_corners, cf_o2_bbox_corners):

            # Check xz
            o1_xz_center = cf_o1_bbox_corners[[0,2], 8] # [x,z]

            # Get camera-frame axis-aligned bbox corners for o2. Only for x-z plane, and two left-most corners. 
            o2_upper_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[2,:8].max()]) # [x,z]
            o2_lower_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[2,:8].min()]) # [x,z]

            # Upper half-space defined by p'n + d = 0
            upper_normal = rotate_2d(np.array([-1,0]), np.pi/2. - self.params['block_theta'])
            upper_d = -1 * o2_upper_corner.dot(upper_normal)
            first_rule = o1_xz_center.dot(upper_normal) + upper_d >= 0

            # Lower half-space defined by p'n + d = 0
            lower_normal = rotate_2d(np.array([0,1]), self.params['block_theta'])
            lower_d = -1 * o2_lower_corner.dot(lower_normal)
            second_rule = o1_xz_center.dot(lower_normal) + lower_d >= 0

            xz_works = first_rule and second_rule

            # Check xy
            o1_xy_center = cf_o1_bbox_corners[[0,1], 8] # [x,y]

            # Get camera-frame axis-aligned bbox corners for o2. Only for x-y plane, and two left-most corners. 
            o2_upper_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[1,:8].max()]) # [x,y]
            o2_lower_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[1,:8].min()]) # [x,y]

            # Upper half-space defined by p'n + d = 0
            upper_normal = rotate_2d(np.array([-1,0]), np.pi/2. - self.params['block_theta'])
            upper_d = -1 * o2_upper_corner.dot(upper_normal)
            third_rule = o1_xy_center.dot(upper_normal) + upper_d >= 0

            # Lower half-space defined by p'n + d = 0
            lower_normal = rotate_2d(np.array([0,1]), self.params['block_theta'])
            lower_d = -1 * o2_lower_corner.dot(lower_normal)
            fourth_rule = o1_xy_center.dot(lower_normal) + lower_d >= 0

            xy_works = third_rule and fourth_rule

            # All corners check
            fifth_rule = np.all(o1_xz_center[0] <= cf_o2_bbox_corners[0,:8].min())

            # o1 right corners check
            sixth_rule = np.all(cf_o1_bbox_corners[0, :8].max() <= cf_o2_bbox_corners[0,8])

            return xz_works and xy_works and fifth_rule and sixth_rule

        obj1_id = o1_predicate_args['obj_id']
        obj2_id = o2_predicate_args['obj_id']

        # For symmetry, check if o1 is left of o2, and if o2 is right of o1
        cf_o1_bbox_corners = o1_predicate_args['cf_bbox_corners'].copy()
        cf_o2_bbox_corners = o2_predicate_args['cf_bbox_corners'].copy()
        o1_left_of_o2 = left_of(cf_o1_bbox_corners, cf_o2_bbox_corners)

        o2_left_of_o1 = left_of(cf_o2_bbox_corners, cf_o1_bbox_corners)

        cf_o1_bbox_corners[0,:] = cf_o1_bbox_corners[0,:] * -1
        cf_o2_bbox_corners[0,:] = cf_o2_bbox_corners[0,:] * -1
        o2_right_of_o1 = left_of(cf_o2_bbox_corners, cf_o1_bbox_corners)

        o1_right_of_o2 = left_of(cf_o1_bbox_corners, cf_o2_bbox_corners)

        if o1_left_of_o2 or o2_right_of_o1:
            # predicates.append(1)
            predicates.append(0)  
        elif o2_left_of_o1 or o1_right_of_o2:
            # predicates.append(0)
            predicates.append(1)  
        else:
            # predicates.append(0)
            predicates.append(0)                

    
    def compute_right_only_predicates(self, o1_predicate_args, o2_predicate_args, predicates):
        """ Compute left-right predicates.
            Use camera frame coordinates.
            Relation rules:
                1) o1 center MUST be in half-space defined by o2 UPPER corner and theta (xz plane)
                2) o1 center MUST be in half-space defined by o2 LOWER corner and theta (xz plane)
                3) do same as 1) for xy
                4) do same as 2) for xy
                5) o1 center MUST be to left of all o2 corners
                6) All o1 corners MUST be to the left of o2 center
        """

        def left_of(cf_o1_bbox_corners, cf_o2_bbox_corners):

            # Check xz
            o1_xz_center = cf_o1_bbox_corners[[0,2], 8] # [x,z]

            # Get camera-frame axis-aligned bbox corners for o2. Only for x-z plane, and two left-most corners. 
            o2_upper_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[2,:8].max()]) # [x,z]
            o2_lower_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[2,:8].min()]) # [x,z]

            # Upper half-space defined by p'n + d = 0
            upper_normal = rotate_2d(np.array([-1,0]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            upper_d = -1 * o2_upper_corner.dot(upper_normal)
            first_rule = o1_xz_center.dot(upper_normal) + upper_d >= 0

            # Lower half-space defined by p'n + d = 0
            lower_normal = rotate_2d(np.array([0,1]), self.params['theta_predicte_lr_fb_ab'])
            lower_d = -1 * o2_lower_corner.dot(lower_normal)
            second_rule = o1_xz_center.dot(lower_normal) + lower_d >= 0

            xz_works = first_rule and second_rule

            # Check xy
            o1_xy_center = cf_o1_bbox_corners[[0,1], 8] # [x,y]

            # Get camera-frame axis-aligned bbox corners for o2. Only for x-y plane, and two left-most corners. 
            o2_upper_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[1,:8].max()]) # [x,y]
            o2_lower_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[1,:8].min()]) # [x,y]

            # Upper half-space defined by p'n + d = 0
            upper_normal = rotate_2d(np.array([-1,0]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            upper_d = -1 * o2_upper_corner.dot(upper_normal)
            third_rule = o1_xy_center.dot(upper_normal) + upper_d >= 0

            # Lower half-space defined by p'n + d = 0
            lower_normal = rotate_2d(np.array([0,1]), self.params['theta_predicte_lr_fb_ab'])
            lower_d = -1 * o2_lower_corner.dot(lower_normal)
            fourth_rule = o1_xy_center.dot(lower_normal) + lower_d >= 0

            xy_works = third_rule and fourth_rule

            # All corners check
            fifth_rule = np.all(o1_xz_center[0] <= cf_o2_bbox_corners[0,:8].min())

            # o1 right corners check
            sixth_rule = np.all(cf_o1_bbox_corners[0, :8].max() <= cf_o2_bbox_corners[0,8])

            return xz_works and xy_works and fifth_rule and sixth_rule

        obj1_id = o1_predicate_args['obj_id']
        obj2_id = o2_predicate_args['obj_id']

        # For symmetry, check if o1 is left of o2, and if o2 is right of o1
        cf_o1_bbox_corners = o1_predicate_args['cf_bbox_corners'].copy()
        cf_o2_bbox_corners = o2_predicate_args['cf_bbox_corners'].copy()
        o1_left_of_o2 = left_of(cf_o1_bbox_corners, cf_o2_bbox_corners)

        o2_left_of_o1 = left_of(cf_o2_bbox_corners, cf_o1_bbox_corners)

        cf_o1_bbox_corners[0,:] = cf_o1_bbox_corners[0,:] * -1
        cf_o2_bbox_corners[0,:] = cf_o2_bbox_corners[0,:] * -1
        o2_right_of_o1 = left_of(cf_o2_bbox_corners, cf_o1_bbox_corners)

        o1_right_of_o2 = left_of(cf_o1_bbox_corners, cf_o2_bbox_corners)

        if o1_left_of_o2 or o2_right_of_o1:
            # predicates.append(1)
            predicates.append(0)  
        elif o2_left_of_o1 or o1_right_of_o2:
            # predicates.append(0)
            predicates.append(1)  
        else:
            # predicates.append(0)
            predicates.append(0)                

    
    def compute_front_behind_predicates(self, o1_predicate_args, o2_predicate_args, predicates):
        """ Compute front-behind predicates.
            Use camera frame coordinates.
            Relation rules:
                1) o1 center MUST be in half-space defined by o2 LEFT corner and theta (xz plane)
                2) o1 center MUST be in half-space defined by o2 RIGHT corner and theta (xz plane)
                3) do same as 1) for yz
                4) do same as 2) for yz
                5) o1 center MUST be behind all o2 corners
                6) All o1 corners MUST be behind o2 center
        """

        def behind(cf_o1_bbox_corners, cf_o2_bbox_corners):

            # Check xz
            o1_xz_center = cf_o1_bbox_corners[[0,2], 8] # [x,z]

            # Get camera-frame axis-aligned bbox corners for o2. Only for x-z plane, and two left-most corners. 
            o2_left_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[2,:8].max()]) # [x,z]
            o2_right_corner = np.array([cf_o2_bbox_corners[0,:8].max(), cf_o2_bbox_corners[2,:8].max()]) # [x,z]

            # Left half-space defined by p'n + d = 0
            left_normal = rotate_2d(np.array([1,0]), self.params['theta_predicte_lr_fb_ab'])
            left_d = -1 * o2_left_corner.dot(left_normal)
            first_rule = o1_xz_center.dot(left_normal) + left_d >= 0

            # Right half-space defined by p'n + d = 0
            right_normal = rotate_2d(np.array([0,1]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            right_d = -1 * o2_right_corner.dot(right_normal)
            second_rule = o1_xz_center.dot(right_normal) + right_d >= 0

            xz_works = first_rule and second_rule

            # Check yz
            o1_yz_center = cf_o1_bbox_corners[[1,2], 8] # [y,z]

            # Get camera-frame axis-aligned bbox corners for o2. Only for x-z plane, and two left-most corners. 
            o2_left_corner = np.array([cf_o2_bbox_corners[1,:8].min(), cf_o2_bbox_corners[2,:8].max()]) # [y,z]
            o2_right_corner = np.array([cf_o2_bbox_corners[1,:8].max(), cf_o2_bbox_corners[2,:8].max()]) # [y,z]

            # Left half-space defined by p'n + d = 0
            left_normal = rotate_2d(np.array([1,0]), self.params['theta_predicte_lr_fb_ab'])
            left_d = -1 * o2_left_corner.dot(left_normal)
            third_rule = o1_yz_center.dot(left_normal) + left_d >= 0

            # Right half-space defined by p'n + d = 0
            right_normal = rotate_2d(np.array([0,1]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            right_d = -1 * o2_right_corner.dot(right_normal)
            fourth_rule = o1_yz_center.dot(right_normal) + right_d >= 0

            yz_works = third_rule and fourth_rule

            # All corners check
            fifth_rule = np.all(o1_xz_center[1] >= cf_o2_bbox_corners[2,:8].max())

            # o1 near corners check
            sixth_rule = np.all(cf_o1_bbox_corners[2, :8].min() >= cf_o2_bbox_corners[2,8])

            return xz_works and yz_works and fifth_rule and sixth_rule

        obj1_id = o1_predicate_args['obj_id']
        obj2_id = o2_predicate_args['obj_id']

        # For symmetry, check if o1 is behind of o2, and if o2 is in front of o1
        cf_o1_bbox_corners = o1_predicate_args['cf_bbox_corners'].copy()
        cf_o2_bbox_corners = o2_predicate_args['cf_bbox_corners'].copy()
        o1_behind_o2 = behind(cf_o1_bbox_corners, cf_o2_bbox_corners)

        o2_behind_o1 = behind(cf_o2_bbox_corners, cf_o1_bbox_corners)

        cf_o1_bbox_corners[2,:] = cf_o1_bbox_corners[2,:] * -1
        cf_o2_bbox_corners[2,:] = cf_o2_bbox_corners[2,:] * -1
        o2_in_front_of_o1 = behind(cf_o2_bbox_corners, cf_o1_bbox_corners) 

        o1_in_front_of_o2 = behind(cf_o1_bbox_corners, cf_o2_bbox_corners)               

        if o1_behind_o2 or o2_in_front_of_o1:
            predicates.append(0)
            predicates.append(1)
        elif o2_behind_o1 or o1_in_front_of_o2:
            predicates.append(1)
            predicates.append(0)
        else:
            predicates.append(0)
            predicates.append(0)

    def compute_above_below_predicates(self, o1_predicate_args, o2_predicate_args, predicates):
        """ Compute above-below predicates.
            Use camera frame coordinates.
            Relation rules:
                1) o1 center MUST be in half-space defined by o2 LEFT corner and theta (xy plane)
                2) o1 center MUST be in half-space defined by o2 RIGHT corner and theta (xy plane)
                3) do same as 1) for zy
                4) do same as 2) for zy
                6) o1 center MUST be above all o2 corners
                7) All o1 corners MUST be above o2 center
                rule = ((1 & 2 & 3 & 4)) & 6 & 7
        """

        def above(cf_o1_bbox_corners, cf_o2_bbox_corners):

            # Check xy
            o1_xy_center = cf_o1_bbox_corners[[0,1], 8] # [x,y]

            # Get camera-frame axis-aligned bbox corners for o2
            o2_left_corner = np.array([cf_o2_bbox_corners[0,:8].min(), cf_o2_bbox_corners[1,:8].max()]) # [x,y]
            o2_right_corner = np.array([cf_o2_bbox_corners[0,:8].max(), cf_o2_bbox_corners[1,:8].max()]) # [x,y]

            # Left half-space defined by p'n + d = 0
            left_normal = rotate_2d(np.array([1,0]), self.params['theta_predicte_lr_fb_ab'])
            left_d = -1 * o2_left_corner.dot(left_normal)
            first_rule = o1_xy_center.dot(left_normal) + left_d >= 0

            # Right half-space defined by p'n + d = 0
            right_normal = rotate_2d(np.array([0,1]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            right_d = -1 * o2_right_corner.dot(right_normal)
            second_rule = o1_xy_center.dot(right_normal) + right_d >= 0

            xy_works = first_rule and second_rule

            # Check zy
            o1_zy_center = cf_o1_bbox_corners[[2,1], 8] # [z,y]

            # Get camera-frame axis-aligned bbox corners for o2 
            o2_left_corner = np.array([cf_o2_bbox_corners[2,:8].min(), cf_o2_bbox_corners[1,:8].max()]) # [z,y]
            o2_right_corner = np.array([cf_o2_bbox_corners[2,:8].max(), cf_o2_bbox_corners[1,:8].max()]) # [z,y]

            # Left half-space defined by p'n + d = 0
            left_normal = rotate_2d(np.array([1,0]), self.params['theta_predicte_lr_fb_ab'])
            left_d = -1 * o2_left_corner.dot(left_normal)
            third_rule = o1_zy_center.dot(left_normal) + left_d >= 0

            # Right half-space defined by p'n + d = 0
            right_normal = rotate_2d(np.array([0,1]), np.pi/2. - self.params['theta_predicte_lr_fb_ab'])
            right_d = -1 * o2_right_corner.dot(right_normal)
            fourth_rule = o1_zy_center.dot(right_normal) + right_d >= 0

            zy_works = third_rule and fourth_rule

            # All corners check
            fifth_rule = np.all(o1_xy_center[1] >= cf_o2_bbox_corners[1,:8].max())

            # o1 bottom corners check
            sixth_rule = np.all(cf_o1_bbox_corners[1, :8].min() >= cf_o2_bbox_corners[1,8])

            return xy_works and zy_works and fifth_rule and sixth_rule

        obj1_id = o1_predicate_args['obj_id']
        obj2_id = o2_predicate_args['obj_id']

        # For symmetry, check if o1 is above o2, and if o2 is below o1
        cf_o1_bbox_corners = o1_predicate_args['cf_bbox_corners'].copy()
        cf_o2_bbox_corners = o2_predicate_args['cf_bbox_corners'].copy()
        o1_above_o2 = above(cf_o1_bbox_corners, cf_o2_bbox_corners)
        o2_above_o1 = above(cf_o2_bbox_corners, cf_o1_bbox_corners)

        cf_o1_bbox_corners[1,:] = cf_o1_bbox_corners[1,:] * -1
        cf_o2_bbox_corners[1,:] = cf_o2_bbox_corners[1,:] * -1
        o2_below_o1 = above(cf_o2_bbox_corners, cf_o1_bbox_corners)    
        o1_below_o2 = above(cf_o1_bbox_corners, cf_o2_bbox_corners)        

        if o1_above_o2 or o2_below_o1:
            # predicates.append((obj1_id, obj2_id, 'above'))
            # predicates.append((obj2_id, obj1_id, 'below'))
            predicates.append(0)
            predicates.append(1)
        elif o2_above_o1 or o1_below_o2:
            predicates.append(1)
            predicates.append(0)
        else:
            predicates.append(0)
            predicates.append(0)

    
    def get_point_cloud_center(self, v):
        
        A = np.min(v[:, :], axis = 0) + (np.max(v[:, :], axis = 0) - np.min(v[:, :], axis = 0))/2
        A_1 = [A[1], A[2], A[0]]

        return np.array(A_1)
    

    def __len__(self):
        return len(self.obj_voxels_by_obj_pair_dict)
    
    
    
    def get_all_object_pair_voxels_fast_3steps(self):

        return self.all_relation_fast, self.obj_voxels_single_fast, self.one_hot_encoding_tensor_fast_batch, self.total_objects_fast, self.edge_attr_fast, self.all_action_fast, self.all_action_label_fast, self.env_identity_list_fast, self.current_pose_fast, self.all_hidden_label_list, self.all_obj_boundary_list_fast, self.current_position, self.current_quaternian, self.all_gt_extents_list_tensor, self.all_hidden_label_fast, self.new_latent, self.support_surface_id_fast, self.buffer_tensor_0, self.all_gt_grapable_list_fast

    def put_all_things_on_device(self, device):
        self.all_relation_fast = self.all_relation_fast.to(device)
        self.obj_voxels_single_fast = self.obj_voxels_single_fast.to(device)
        self.one_hot_encoding_tensor_fast_batch = self.one_hot_encoding_tensor_fast_batch.to(device)
        self.edge_attr_fast = self.edge_attr_fast.to(device)
        self.all_action_fast = self.all_action_fast.to(device)
        self.env_identity_list_fast = self.env_identity_list_fast.to(device)
        self.all_gt_grapable_list_fast = self.all_gt_grapable_list_fast.to(device)
        self.current_pose_fast = self.current_pose_fast.to(device)
        self.all_obj_boundary_list_fast = self.all_obj_boundary_list_fast.to(device)
        self.all_hidden_label_fast = self.all_hidden_label_fast.to(device)
        self.new_latent = self.new_latent.to(device)
        
        self.current_position = self.current_position.to(device)
        self.current_quaternian = self.current_quaternian.to(device)
        self.all_gt_extents_list_tensor = self.all_gt_extents_list_tensor.to(device)

        for each_id in range(len(self.support_surface_id_fast)):
            self.support_surface_id_fast[each_id] = self.support_surface_id_fast[each_id].to(device)
        self.buffer_tensor_0 = self.buffer_tensor_0.to(device)

        
    def get_obj_num(self):
        return self.total_objects
    
class DataLoader(object):
    def __init__(self, 
                 config,
                 relation_angle, 
                 max_objects = 5, 
                 use_multiple_train_dataset = False,
                 use_multiple_test_dataset = False,
                 pick_place = False,
                 pushing = False,
                 stacking = False, 
                 set_max = False,
                 train_dir_list=None,
                 test_dir_list=None,
                 load_contact_data=False,
                 start_id = 0, 
                 max_size = 0,   # begin on max_size = 8000 for all disturbance data
                 start_test_id = 0, 
                 test_max_size = 2,
                 updated_behavior_params = False,
                 save_data_path = None, 
                 evaluate_new = False, 
                 using_multi_step_statistics = False,
                 total_multi_steps = 0,
                 use_shared_latent_embedding = False,
                 use_seperate_latent_embedding = False,
                 online_planning = False,
                 train_object_identity = False,
                 use_rgb = False,
                 push_3_steps = False,
                 use_boundary_relations = False,
                 consider_z_offset = False,
                 seperate_env_id = False,
                 max_env_num = False,
                 env_first_step = False,
                 use_discrete_z = False,
                 fast_training = False,
                 evaluate_pickplace = False,
                 one_bit_env = False,
                 rcpe = False,
                 pe = False, 
                 bookshelf_env_shift = 0,
                 push_steps = 2, 
                 POMDP_push = False,
                 sudo_pickplace = False, 
                 add_noise_pc = False, 
                 enable_return = False,
                 lfd_search = False,
                 test_data_loader = False,
                 get_hidden_label = False,
                 get_inside_relations = False,
                 enable_place_inside = False,
                 binary_grasp = False,
                 open_close_drawer = False, 
                 softmax_identity = False,
                 train_inside_feasibility = False, 
                 use_discrete_place = False,
                 seperate_place = False,
                 enable_leap_num = False,
                 single_step_training = False,
                 batch_feasibility = False,
                 ):
        #self.train = train
        
        self.test_data_loader = test_data_loader
        self.lfd_search = lfd_search
        self.enable_return = enable_return
        
        self.push_steps = push_steps
        self.total_multi_steps = total_multi_steps
        self.using_multi_step_statistics = using_multi_step_statistics
        self.evaluate_new = evaluate_new
        self.evaluate_pickplace = evaluate_pickplace
        self.updated_behavior_params = updated_behavior_params
        stacking = stacking
        self.set_max = set_max
        self._config = config
        self.transforms = None
        self.pos_grid = None
        self.pick_place = pick_place
        self.pushing = pushing
        self.stacking = stacking
        
        self.load_contact_data = load_contact_data
        self.use_shared_latent_embedding = use_shared_latent_embedding
        self.use_seperate_latent_embedding = use_seperate_latent_embedding
        self.online_planning = online_planning
        self.push_3_steps = push_3_steps
        
        self.sudo_pickplace = sudo_pickplace
        self.POMDP_push = POMDP_push
        
        self.add_noise_pc = add_noise_pc
        
        self.train_object_identity = train_object_identity
        self.use_rgb = use_rgb
        self.use_boundary_relations = use_boundary_relations
        self.consider_z_offset = consider_z_offset
        self.seperate_env_id = seperate_env_id
        self.env_first_step = env_first_step
        self.max_env_num = max_env_num
        self.use_discrete_z = use_discrete_z
        self.fast_training = fast_training
        self.rcpe = rcpe
        self.pe = pe
        self.relation_angle = relation_angle
        self.get_hidden_label = get_hidden_label
        self.enable_leap_num = enable_leap_num

        self.one_bit_env = one_bit_env
        self.bookshelf_env_shift = bookshelf_env_shift

        self.fail_reasoning_num = 0

        self.get_inside_relations = get_inside_relations

        self.enable_place_inside = enable_place_inside

        self.binary_grasp = binary_grasp

        self.open_close_drawer = open_close_drawer

        self.softmax_identity = softmax_identity

        self.train_inside_feasibility = train_inside_feasibility

        self.use_discrete_place = use_discrete_place

        self.seperate_place = seperate_place

        self.scene_type = "data_in_line"

        self.single_step_training = single_step_training

        self.batch_feasibility = batch_feasibility

        if self.sudo_pickplace:
            total_steps = 2


        

        demo_idx = 0
        self.train_idx_to_data_dict = {}
        idx_to_data_dict = {}


        data_size = 128
        self.max_size = max_size
        self.test_max_size = test_max_size

        
        
        
        self.motion_planner_fail_num = 0
        self.train_id = 0
        self.test_id = 0   

        self.all_goal_relations = np.ones((50000,5,1))
        self.all_predicted_relations = np.ones((50000,5,1))
        self.all_index_i_list = np.ones((50000,5,1))
        self.all_index_j_list = np.ones((50000,5,1))
        self.all_planning_failure_list = np.ones((50000,1)) 

        if config.args.train_dir != None:
            self.train_dir_list = train_dir_list \
                if train_dir_list is not None else config.args.train_dir

            
            
            files = sorted(os.listdir(self.train_dir_list[0]))       
            self.train_pcd_path = [
                os.path.join(self.train_dir_list[0], p) for p in files if 'demo' in p]
            for train_dir in self.train_pcd_path[start_id:start_id+max_size]:
                self.current_goal_relations = self.all_goal_relations[self.train_id] # simplify for without test end_relations
                self.current_predicted_relations = self.all_predicted_relations[self.train_id]
                self.current_index_i = self.all_index_i_list[self.train_id]  ## test_id?  I change it to train_id
                self.current_index_j = self.all_index_j_list[self.train_id]
                self.train_id += 1
 
                with open(train_dir, 'rb') as f:
                    data, attrs = pickle.load(f)
                
                print('loaded data:', train_dir)    
                total_objects = 0
                for k, v in data.items():
                    if 'point_cloud' in k and 'sampling' in k and 'last' not in k:
                        total_objects += 1
                this_one_hot_encoding = np.zeros((1, total_objects))
                
                leap = 1

                
                if True:
                    for k, v in data.items():
                        if 'point_cloud' in k and 'last' not in k and 'leap' not in k:
                            if(v.shape[0] == 0):
                                leap = 0
                                break
                            if(v.shape[0] != total_steps) and not self.sudo_pickplace:
                                leap = 0
                                break
                            for i in range((v.shape[0])):
                                if(v[i].shape[0] < data_size):
                                    leap = 0
                                    break

                        

                if leap == 0:
                    continue
                
                if self.single_step_training:

                    for each_step in range(data['point_cloud_1'].shape[0] - 1):
                        self.train = True

                        all_pair_scene_object =  PerSceneLoader(train_dir, self.scene_type, start = each_step, end = each_step + 2, single_step_training = self.single_step_training, pick_place = self.pick_place, push = self.pushing, set_max = self.set_max, max_objects = max_objects ,train = self.train, updated_behavior_params = self.updated_behavior_params, use_shared_latent_embedding = self.use_shared_latent_embedding, push_3_steps = self.push_3_steps, use_seperate_latent_embedding = self.use_seperate_latent_embedding, train_object_identity = self.train_object_identity, use_boundary_relations = self.use_boundary_relations, consider_z_offset = self.consider_z_offset, seperate_env_id = self.seperate_env_id, max_env_num = self.max_env_num, env_first_step = self.env_first_step, use_discrete_z = self.use_discrete_z, fast_training = self.fast_training, one_bit_env = self.one_bit_env, rcpe = self.rcpe, pe = self.pe, relation_angle = self.relation_angle, bookshelf_env_shift = self.bookshelf_env_shift, push_steps = self.push_steps ,enable_return = self.enable_return, lfd_search = self.lfd_search, get_hidden_label = self.get_hidden_label, get_inside_relations = self.get_inside_relations, enable_place_inside = self.enable_place_inside, binary_grasp = self.binary_grasp, open_close_drawer = self.open_close_drawer, softmax_identity = self.softmax_identity, train_inside_feasibility = self.train_inside_feasibility, use_discrete_place = self.use_discrete_place, seperate_place = self.seperate_place, add_noise_pc = self.add_noise_pc, sudo_pickplace = self.sudo_pickplace)
                        
                        
                        if all_pair_scene_object.data_filter_leap == 1:
                            idx_to_data_dict[demo_idx] = {}
                        
                            idx_to_data_dict[demo_idx]['objects'] = data['objects']

                            idx_to_data_dict[demo_idx]['att_objects'] = attrs['objects']

                            idx_to_data_dict[demo_idx]['goal_relations'] = self.current_goal_relations
                            idx_to_data_dict[demo_idx]['predicted_relations'] = self.current_predicted_relations
                            idx_to_data_dict[demo_idx]['index_i'] = self.current_index_i
                            idx_to_data_dict[demo_idx]['index_j'] = self.current_index_j
                            idx_to_data_dict[demo_idx]['this_one_hot_encoding'] = this_one_hot_encoding

                            total_point_cloud = np.concatenate((data['point_cloud_1'][0][:data_size][:], data['point_cloud_2'][0][:data_size][:]), axis = 0)
                            idx_to_data_dict[demo_idx]['point_cloud'] = total_point_cloud.T
                            
                            idx_to_data_dict[demo_idx]['path'] = train_dir
                            idx_to_data_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
                            demo_idx += 1           
                
            self.train_idx_to_data_dict.update(idx_to_data_dict)
                    

    
        if len(self.train_idx_to_data_dict) > 0:
            print("Did create index for train data: {}".format(
                len(self.train_idx_to_data_dict)))

        self.test_idx_to_data_dict = {}
        if config.args.test_dir != None:
            self.test_dir_list = test_dir_list \
                if test_dir_list is not None else config.args.test_dir
            
            idx_to_data_dict = {}

            files = sorted(os.listdir(self.test_dir_list[0]))
            
            self.test_pcd_path = [
                os.path.join(self.test_dir_list[0], p) for p in files if 'demo' in p]

            if self.enable_leap_num:
                test_max_size = 1000
                self.success_load = 0
            for test_dir in self.test_pcd_path[start_test_id: start_test_id + test_max_size]:
                if self.enable_leap_num:
                    if self.success_load == self.test_max_size:
                        break
                if self.evaluate_new:
                    self.current_task_plan = self.all_task_planner[self.train_id]
                    self.current_goal_relations = self.all_goal_relations[self.train_id] # simplify for without test end_relations
                    self.current_predicted_relations = self.all_predicted_relations[self.train_id]
                    self.current_index_i = self.all_index_i_list[self.train_id]
                    self.current_index_j = self.all_index_j_list[self.train_id]
                    planning_success = 1
                    if self.train_id == 0:
                        if self.all_planning_failure_list[self.train_id] != 0:
                            planning_success = 0
                    else:
                        if self.all_planning_failure_list[self.train_id] - self.all_planning_failure_list[self.train_id - 1] != 0:
                            planning_success = 0

                else:
                    self.current_goal_relations = self.all_goal_relations[0] # simplify for without test end_relations
                    self.current_predicted_relations = self.all_goal_relations[0] # simplify for without test end_relations
                self.train_id += 1
                       
                with open(test_dir, 'rb') as f:
                    data, attrs = pickle.load(f)
                print('loaded data:', test_dir)     
                leap = 1

                
                for k, v in data.items():
                    if 'point_cloud' in k and 'last' not in k:
                        if True:
                            if(v.shape[0] == 0):
                                leap = 0
                                break
                            
                            if(v.shape[0] != total_steps) and not self.sudo_pickplace: # 2 will be total steps of the task
                                leap = 0
                                break
                            for i in range((v.shape[0])):   ## filter out some examples that have too fewer points, for example, object fall over the table. 
                                if(v[i].shape[0] < data_size):
                                    leap = 0
                                    break
                      
                    
                if leap == 0:
                    continue
                if self.enable_leap_num:
                    self.success_load += 1
                
                
                self.train = False
                    
                all_pair_scene_object =  PerSceneLoader(test_dir, self.scene_type, pick_place = self.pick_place, push = self.pushing, set_max = self.set_max, max_objects = max_objects ,train = self.train, updated_behavior_params = self.updated_behavior_params, use_shared_latent_embedding = self.use_shared_latent_embedding, push_3_steps = self.push_3_steps, use_seperate_latent_embedding = self.use_seperate_latent_embedding, train_object_identity = self.train_object_identity, use_boundary_relations = self.use_boundary_relations, consider_z_offset = self.consider_z_offset, seperate_env_id = self.seperate_env_id, max_env_num = self.max_env_num, env_first_step = self.env_first_step, use_discrete_z = self.use_discrete_z, fast_training = self.fast_training, one_bit_env = self.one_bit_env, rcpe = self.rcpe, pe = self.pe, relation_angle = self.relation_angle, bookshelf_env_shift = self.bookshelf_env_shift, push_steps = self.push_steps ,enable_return = self.enable_return, lfd_search = self.lfd_search, POMDP_push = self.POMDP_push, add_noise_pc = self.add_noise_pc, get_hidden_label = self.get_hidden_label, get_inside_relations = self.get_inside_relations, enable_place_inside = self.enable_place_inside, binary_grasp = self.binary_grasp, open_close_drawer = self.open_close_drawer, softmax_identity = self.softmax_identity, train_inside_feasibility = self.train_inside_feasibility, use_discrete_place = self.use_discrete_place, seperate_place = self.seperate_place, sudo_pickplace = self.sudo_pickplace)
                    
                if self.evaluate_new and planning_success == 0:
                    self.planning_failure_num += 1
                elif all_pair_scene_object.data_filter_leap == 1 or not self.evaluate_new:
                    idx_to_data_dict[demo_idx] = {}
                    
                    idx_to_data_dict[demo_idx]['objects'] = data['objects']

                    idx_to_data_dict[demo_idx]['att_objects'] = attrs['objects']

                    
                    this_one_hot_encoding = np.zeros((1, 3)) ## quick test
                    idx_to_data_dict[demo_idx]['this_one_hot_encoding'] = this_one_hot_encoding
                    

                    if self.evaluate_new:
                        idx_to_data_dict[demo_idx]['goal_relations'] = self.current_goal_relations
                        idx_to_data_dict[demo_idx]['predicted_relations'] = self.current_predicted_relations
                        idx_to_data_dict[demo_idx]['index_i'] = self.current_index_i
                        idx_to_data_dict[demo_idx]['index_j'] = self.current_index_j
                        idx_to_data_dict[demo_idx]['task_planner'] = self.current_task_plan
                    else:
                        idx_to_data_dict[demo_idx]['goal_relations'] = self.current_goal_relations
                        idx_to_data_dict[demo_idx]['predicted_relations'] = self.current_predicted_relations
                        idx_to_data_dict[demo_idx]['index_i'] = [1]
                        idx_to_data_dict[demo_idx]['index_j'] = [1]

                    
                    
                    total_point_cloud = np.concatenate((data['point_cloud_1'][0][:data_size][:], data['point_cloud_2'][0][:data_size][:]), axis = 0)
                    
                    
                    idx_to_data_dict[demo_idx]['path'] = test_dir
                    idx_to_data_dict[demo_idx]['scene_voxel_obj'] = all_pair_scene_object
                    demo_idx += 1           
                else:
                    if self.evaluate_new:
                        self.fail_data_filter_num += 1               
            self.test_idx_to_data_dict.update(idx_to_data_dict)

            
            print("Did create index for test data: {}".format(
                len(self.test_idx_to_data_dict)))


        self.train_scene_sample_order = {}
        self.test_scene_sample_order = {}

    def get_fail_reasoning_num(self):
        return self.fail_reasoning_num

    def get_fail_leap_num(self):
        return self.fail_data_filter_num

    def get_fail_planning_num(self):
        return self.planning_failure_num
    

    
    def get_demo_data_dict(self, train=True):
        data_dict = self.train_idx_to_data_dict if train else self.test_idx_to_data_dict
        return data_dict
    
    
    def number_of_scenes(self, train=True):
        data_dict = self.get_demo_data_dict(train)
        return len(data_dict)

    def reset_scene_batch_sampler(self, train=True, shuffle=True) -> None:
        if train:
            sampling_dict = self.train_scene_sample_order
        elif not train:
            sampling_dict = self.test_scene_sample_order
        else:
            raise ValueError("Invalid value")

        data_dict = self.get_demo_data_dict(train)
        order = sorted(data_dict.keys())
        if shuffle:
            np.random.shuffle(order)


        sampling_dict['order'] = order
        sampling_dict['idx'] = 0
        


    def number_of_scene_data(self, train=True):
        return self.number_of_scenes(train)

    


    def get_all_object_pairs_for_scene_index(self, scene_idx, train=True):
        data_dict = self.get_demo_data_dict(train)[scene_idx]
 
        path = data_dict['path']
        scene_voxel_obj = data_dict['scene_voxel_obj']
        
        # we need to make sure all the following return arguments have the same value and structure. 
        all_relation_fast, obj_voxels_single_fast, one_hot_encoding_tensor_fast, total_objects_fast, edge_attr_fast, all_action_fast, all_action_label_fast, all_env_identity_fast, all_6DOF_pose_fast, all_hidden_label_list, all_obj_boundary_list, all_position, all_quaternian, all_extents, all_hidden_tensor, new_latent, support_suface_id, buffer_tensor_0, all_gt_grapable_list = scene_voxel_obj.get_all_object_pair_voxels_fast_3steps()
        
        if self.evaluate_new:
            data = {
                'num_objects': total_objects_fast,
                'action': all_action_fast,
                'relation': all_relation_fast, 
                'all_object_pair_voxels_single': obj_voxels_single_fast,
                'one_hot_encoding': one_hot_encoding_tensor_fast,
                'edge_attr': edge_attr_fast,
                'all_action_label':all_action_label_fast,
                'env_identity': all_env_identity_fast,
                'all_6DOF_pose_fast': all_6DOF_pose_fast,
                'all_hidden_label': all_hidden_label_list, 
                'all_obj_boundaty': all_obj_boundary_list,
                'position': all_position,
                'quaternian': all_quaternian,
                'extents': all_extents,
                'all_hidden_tensor': all_hidden_tensor,
                'new_latent': new_latent,
                'support_suface_id': support_suface_id,
                'buffer_tensor_0': buffer_tensor_0,
                'all_gt_grapable_list': all_gt_grapable_list, 
                'goal_relations': data_dict['goal_relations'],
                'predicted_relations': data_dict['predicted_relations'],
                'index_i': data_dict['index_i'],
                'index_j': data_dict['index_j'],
                'task_planner': data_dict['task_planner']
            }
            data_last = {
                'num_objects': total_objects_fast,
                'action': all_action_fast,
                'relation': all_relation_fast, 
                'all_object_pair_voxels_single': obj_voxels_single_fast,
                'one_hot_encoding': one_hot_encoding_tensor_fast,
                'edge_attr': edge_attr_fast,
                'all_action_label':all_action_label_fast,
                'env_identity': all_env_identity_fast,
                'all_6DOF_pose_fast': all_6DOF_pose_fast,
                'all_hidden_label': all_hidden_label_list,
                'all_obj_boundaty': all_obj_boundary_list,
                'position': all_position,
                'quaternian': all_quaternian,
                'extents': all_extents,
                'all_hidden_tensor': all_hidden_tensor,
                'new_latent': new_latent,
                'support_suface_id': support_suface_id,
                'buffer_tensor_0': buffer_tensor_0,
                'all_gt_grapable_list': all_gt_grapable_list,
                'goal_relations': data_dict['goal_relations'],
                'predicted_relations': data_dict['predicted_relations'],
                'index_i': data_dict['index_i'],
                'index_j': data_dict['index_j'],
                'task_planner': data_dict['task_planner']
            }
        
        else:
            data = {
                'num_objects': total_objects_fast,
                'action': all_action_fast,
                'relation': all_relation_fast, 
                'all_object_pair_voxels_single': obj_voxels_single_fast,
                'one_hot_encoding': one_hot_encoding_tensor_fast,
                'edge_attr': edge_attr_fast,
                'all_action_label':all_action_label_fast,
                'env_identity': all_env_identity_fast,
                'all_6DOF_pose_fast': all_6DOF_pose_fast,
                'all_hidden_label': all_hidden_label_list, 
                'all_obj_boundaty': all_obj_boundary_list,
                'position': all_position,
                'quaternian': all_quaternian,
                'extents': all_extents,
                'all_hidden_tensor': all_hidden_tensor,
                'new_latent': new_latent,
                'support_suface_id': support_suface_id,
                'buffer_tensor_0': buffer_tensor_0,
                'all_gt_grapable_list': all_gt_grapable_list, 
            }
            data_last = {
                'num_objects': total_objects_fast,
                'action': all_action_fast,
                'relation': all_relation_fast, 
                'all_object_pair_voxels_single': obj_voxels_single_fast,
                'one_hot_encoding': one_hot_encoding_tensor_fast,
                'edge_attr': edge_attr_fast,
                'all_action_label':all_action_label_fast,
                'env_identity': all_env_identity_fast,
                'all_6DOF_pose_fast': all_6DOF_pose_fast,
                'all_hidden_label': all_hidden_label_list,
                'all_obj_boundaty': all_obj_boundary_list,
                'position': all_position,
                'quaternian': all_quaternian,
                'extents': all_extents,
                'all_hidden_tensor': all_hidden_tensor,
                'new_latent': new_latent,
                'support_suface_id': support_suface_id,
                'buffer_tensor_0': buffer_tensor_0,
                'all_gt_grapable_list': all_gt_grapable_list
            }
        
        
        return data, data_last
    
    def get_next_all_object_pairs_for_scene(self, train=True):
        # First find the next scene index based on the current index
        sample_order_dict = self.train_scene_sample_order if train else \
            self.test_scene_sample_order
        # Get the current sample pointer
        sample_idx = sample_order_dict['idx']
        # Get the actual scene index.
        scene_idx = sample_order_dict['order'][sample_idx]
        

        data, data_next = self.get_all_object_pairs_for_scene_index(scene_idx, train=train)
        sample_order_dict['idx'] += 1
        
        return data, data_next
       

    def put_all_data_device(self, device):
        for i in range(len(self.train_idx_to_data_dict)):
            data_dict = self.train_idx_to_data_dict[i]['scene_voxel_obj']
            data_dict.put_all_things_on_device(device)

        for i in range(len(self.test_idx_to_data_dict)):
            data_dict = self.test_idx_to_data_dict[i]['scene_voxel_obj']
            data_dict.put_all_things_on_device(device)
