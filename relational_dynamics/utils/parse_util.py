import argparse
from relational_dynamics.utils.data_utils import str2bool

def get_parser():
    parser = argparse.ArgumentParser(
        description='Training/planning for Relational Dynamics.')
    parser.add_argument('--cuda', type=int, default=1, help="Use cuda")
    parser.add_argument('--result_dir', type=str, required=True,
                        help='Directory to save results.')
    parser.add_argument('--checkpoint_path', type=str, default='',
                        help='Checkpoint to test on.')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='Number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for each step')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate for encoder/decoder.')
    parser.add_argument('--train_dir', required=False, action='append',
                        help='Path to training directory.')
    parser.add_argument('--test_dir', required=False, action='append',
                        help='Path to test directory.')
    parser.add_argument('--emb_lr', required=False, type=float, default=0.0001,
                        help='Learning rate to use for pointconvembeddings.')
    parser.add_argument('--z_dim', type=int, default=9,
                        help='number of relations to use.')
    parser.add_argument('--save_data_path', type=str, default='', 
                        help='Path to savetxt file to get goal relations.')
    parser.add_argument('--evaluate_new', type=str2bool, default=False,
                        help='whether to use evaluate_new or not')
    parser.add_argument('--get_tp_success', type=str2bool, default=False,
                        help='whether to use get_tp_success or not')
    parser.add_argument('--set_max', type=str2bool, default=True,
                        help='whether to use set_max method')
    parser.add_argument('--max_objects', type=int, default=8,
                        help='max_objects in this experiments')
    parser.add_argument('--online_planning', type=str2bool, default=False,
                        help='whether to use online_planning in real_data')
    parser.add_argument('--use_multiple_train_dataset', type=str2bool, default=False,
                        help='whether to use use_multiple_train_dataset')
    parser.add_argument('--use_multiple_test_dataset', type=str2bool, default=False,
                        help='whether to use use_multiple_test_dataset')
    parser.add_argument('--evaluate_pickplace', type=str2bool, default=False,
                        help='whether to use evaluate_pickplace')
    parser.add_argument('--updated_behavior_params', type=str2bool, default=True,
                        help='whether to use updated_behavior_params')
    parser.add_argument('--start_id', type=int, default=0,
                        help='start_id in hthe training')
    parser.add_argument('--max_size', type=int, default=0,
                        help='max_size if the training dataset')
    parser.add_argument('--start_test_id', type=int, default=0,
                        help='start_test_id of the test dataset')
    parser.add_argument('--test_max_size', type=int, default=0,
                        help='test_max_size of the test dataset') 
    parser.add_argument('--set_random_seed', type=str2bool, default=False,
                        help='whether to set random seed')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for numpy and torch')
    parser.add_argument('--pick_place', type=str2bool, default=False,
                        help='whether to use pick place skill')        
    parser.add_argument('--pushing', type=str2bool, default=True,
                        help='whether to use pushing skill')  
    parser.add_argument('--rcpe', type=str2bool, default=True,
                        help='whether to use relational classifier and pose estimation baselines') 
    parser.add_argument('--delta_forward', type=str2bool, default=True,
                        help='whether to use delta latent state')   
    parser.add_argument('--latent_forward', type=str2bool, default=False,
                        help='whether to use the latent forward')   
    parser.add_argument('--using_multi_step', type=str2bool, default=False,
                        help='whether to use multi step planning as a task and motion planning style') 
    parser.add_argument('--graph_search', type=str2bool, default=False,
                        help='whether to use graph search in the multi-step planning')
    parser.add_argument('--using_multi_step_statistics', type=str2bool, default=False,
                        help='whether to use using_multi_step_statistics to get statistics for the multi-step test.')
    parser.add_argument('--use_shared_latent_embedding', type=str2bool, default=False,
                        help='whether to use shared latent embedding.')
    parser.add_argument('--use_seperate_latent_embedding', type=str2bool, default=True,
                        help='whether to use seperate latent embedding, which means that we use one latent dynamics for pickplace and one latent dynamics for push.')
    parser.add_argument('--planning_batch_size', type=int, default=3000,
                        help='number of planning_batch_size in the sampling-based planning')
    parser.add_argument('--x_range', type=float, default=0.4,
                        help='x_range sampling in the sampling-based planning')
    parser.add_argument('--y_range', type=float, default=0.4,
                        help='y_range sampling in the sampling-based planning')
    parser.add_argument('--z_range', type=float, default=0.8,
                        help='z_range sampling in the sampling-based planning')
    parser.add_argument('--push_3_steps', type=str2bool, default=True,
                        help='whether to use push_3_steps to deal with multi-step problem.')
    parser.add_argument('--real_online_planning', type=str2bool, default=False,
                        help='whether to use online planning in real-world.')
    parser.add_argument('--train_object_identity', type=str2bool, default=False,
                        help='whether to use object identity.')
    parser.add_argument('--train_env_identity', type=str2bool, default=True,
                        help='whether to use env identity, for example, identity env as 0 and all other objects as 1')
    parser.add_argument('--train_grasp_identity', type=str2bool, default=True,
                        help='whether to use grasp identity, for example, identity graspable as 0 and all other objects as 1')
    parser.add_argument('--train_inside_feasibility', type=str2bool, default=True,
                        help='whether to use inside identity')
    parser.add_argument('--train_obj_boundary', type=str2bool, default=False,
                        help='whether to get object boundary during training and test')
    parser.add_argument('--use_rgb', type=str2bool, default=False,
                        help='whether to use rgb channels as the input.')
    parser.add_argument('--use_boundary_relations', type=str2bool, default=True,
                        help='whether to use boundary relations.') 
    parser.add_argument('--consider_z_offset', type=str2bool, default=False,
                        help='whether to consider_z_offset.')
    parser.add_argument('--seperate_env_id', type=str2bool, default=False,
                        help='whether to seperate env id and object id.')
    parser.add_argument('--max_env_num', type=int, default=0,
                        help='the max number for the environment nodes.')
    parser.add_argument('--seperate_action_emb', type=str2bool, default=True,
                        help='whether to use different action embed based on the different skills.')
    parser.add_argument('--seperate_discrete_continuous', type=str2bool, default=True,
                        help='whether to seperate_discrete_continuous or not.')
    parser.add_argument('--simple_encoding', type=str2bool, default=False,
                        help='whether to simple_encoding or not.')
    parser.add_argument('--env_first_step', type=str2bool, default=False,
                        help='whether to env_first_step or not.')
    parser.add_argument('--use_discrete_z', type=str2bool, default=False,
                        help='whether to use discrete z value for different action in z dimension.')
    parser.add_argument('--transformer_decoder', type=str2bool, default=False,
                        help='whether to use only transformer before decoder.')
    parser.add_argument('--pose_trans_decoder', type=str2bool, default=False,
                        help='whether to use transformer before pose decoder only.')
    parser.add_argument('--transformer_dynamics', type=str2bool, default=True,
                        help='whether to use transformer dynamics or not.')
    parser.add_argument('--use_tensorboard', type=str2bool, default=False,
                        help='whether to use tensorboard in pytorch or not for visualization purpose.')
    parser.add_argument('--torch_embedding', type=str2bool, default=True,
                        help='whether to use torch_embedding or not.')
    parser.add_argument('--complicated_pre_dynamics', type=str2bool, default=True,
                        help='whether to use complicated_pre_dynamics or not.')
    parser.add_argument('--fast_training', type=str2bool, default=True,
                        help='whether to use fast_training or not.')
    parser.add_argument('--fast_training_test', type=str2bool, default=True,
                        help='whether to use fast_training or not.')     
    parser.add_argument('--one_bit_env', type=str2bool, default=True,
                        help='whether to use one bit to represent environment identity.') 
    parser.add_argument('--lfd_search', type=str2bool, default=False,
                        help='whether to use lfd for graph search or not.') 
    parser.add_argument('--direct_transformer', type=str2bool, default=True,
                        help='whether to direct_transformer or not.')
    parser.add_argument('--enable_high_push', type=str2bool, default=False,
                        help='whether to enable_high_push or not.')
    parser.add_argument('--enable_place_inside', type=str2bool, default=False,
                        help='whether to enable_place_inside or not.')
    parser.add_argument('--seperate_place', type=str2bool, default=False,
                        help='whether to seperate_place or not.')
    parser.add_argument('--use_mlp_encoder', type=str2bool, default=False,
                        help='whether to ues use_mlp_encoder or not.')
    parser.add_argument('--pe', type=str2bool, default=True,
                        help='whether to ues pose estimation or not.')
    parser.add_argument('--bookshelf_env_shift', type=int, default=0,
                        help='to solve the shift in segmentation id in bookshelf case. if shift use 1, else use 0')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='the nlayers for transformer')
    parser.add_argument('--get_hidden_label', type=str2bool, default=True,
                        help='whether to get hidden label from dataloader.')
    parser.add_argument('--POMDP_push', type=str2bool, default=True,
                        help='whether to POMDP_push from dataloader.')
    parser.add_argument('--sudo_pickplace', type=str2bool, default=True,
                        help='whether to sudo_pickplace from dataloader.')
    parser.add_argument('--single_step_training', type=str2bool, default=True,
                        help='whether to single_step_training from dataloader.')
    parser.add_argument('--get_inside_relations', type=str2bool, default=True,
                        help='whether to get_inside_relations.')
    parser.add_argument('--remove_orientation', type=str2bool, default=True,
                        help='whether to remove_orientation or not.')
    parser.add_argument('--batch_feasibility', type=str2bool, default=False,
                        help='whether to use batch_feasibility for planner')
    parser.add_argument('--train_obj_move', type=str2bool, default=False,
                        help='whether to use train_obj_move to debug')
    parser.add_argument('--sqrt_loss', type=str2bool, default=True,
                        help='whether to use sqrt_loss to debug')
    parser.add_argument('--current_observation_id', type=int, default=0,
                        help='the current_observation_id for memory model')
    parser.add_argument('--push_steps', type=int, default=2,
                        help='the total push_steps for memory model')
    parser.add_argument('--enable_leap_num', type=str2bool, default=False,
                        help='whether to use enable_leap_num for the dataloader.')
    parser.add_argument('--generate_erd_pickle', type=str2bool, default=False,
                        help='whether to use generate_erd_pickle in during planner.')
    parser.add_argument('--add_noise_pc', type=str2bool, default=False,
                        help='whether to use use add_noise_pc in during dataloader.')
    parser.add_argument('--use_discrete_place', type=str2bool, default=True,
                        help='whether to use use_discrete_place during training/planning.')
    parser.add_argument('--latent_discrete_continuous', type=str2bool, default=True,
                        help='whether to use latent_discrete_continuous during training/planning.')
    parser.add_argument('--binary_grasp', type=str2bool, default=True,
                        help='whether to use binary_grasp during train/planning.')
    parser.add_argument('--open_close_drawer', type=str2bool, default=True,
                        help='whether to use open_close_drawer during train/planning.')
    parser.add_argument('--softmax_identity', type=str2bool, default=False,
                        help='whether to use softmax_identity during train/planning.')
    parser.add_argument('--seperate_identity', type=str2bool, default=False,
                        help='whether to use seperate_identity during train/planning.')
    parser.add_argument('--TP_LLM', type=str2bool, default=False,
                        help='whether to use TP_LLM during planning.')
    parser.add_argument('--sqrt_var',  type=int, default=5,
                        help='the value of sqrt_var.')
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                        help='the dim_feedforward for transformer')
    parser.add_argument('--n_heads', type=int, default=2,
                        help='the nheads for transformer')
    parser.add_argument('--node_emb_size', type=int, default=128,
                        help='the node_emb_size for transformer')
    parser.add_argument('--d_hidden', type=int, default=64,
                        help='the d_hidden for transformer')
    parser.add_argument('--pose_num', type=int, default=2,
                        help='the pose num we use. xyz for 3 and xy for 2')
    parser.add_argument('--y_collision', type=float, default=0.12,
                        help='y_collision value')
    parser.add_argument('--x_collision', type=float, default=0.12,
                        help='x_collision value')
    parser.add_argument('--delta_threshold', type=float, default=0.002,
                        help='delta_threshold value')
    parser.add_argument('--relation_angle', type=float, default=2.25,
                        help='real_relation angle = np.pi/relation_angle')
    parser.add_argument('--total_sub_step', type=int, default=2,
                        help='total sub steps for multi-step test')
    return parser