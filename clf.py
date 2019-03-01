"""
利用已经经过训练的 horse_net_D, zebra_net_D 进行 horse|zebra classification.
"""
from models.discriminate_model import DiscriminateModel
from util.visualizer import Visualizer
from data.discriminator_dataset import horseDataset
from torch.utils.data import DataLoader
import os

class Option():
    pass

if __name__ == '__main__':
    # 1. 创建 options
    opt = Option()
    # todo 
    option_dict = {
        'isTrain': True,
        # BaseModel.__init__()
        'gpu_ids': [0],
        'checkpoints_dir': './checkpoints/zebra',
        'test_results_dir': './test_results/zebra',
        'name': 'experiment1',
        'preprocess': None,
        # model.setup()
        'continue_train': False,  # 虽然这里写的是False, 但是实际上则是 load net from previous_model
        'previous_model': 'checkpoints/horse/best_result/horse_net_D.pth', 'load_iter': 0,
        'epoch': 'latest', 'verbose': True,  # DiscriminateModel.__init__()
        'model_suffix': '', 'output_nc': 3, 'ndf': 64, 'netD': 'basic', 'n_layers_D': 3,
        'norm': 'instance', 'init_type': 'normal', 'init_gain': 0.02, 'display_id': -1,
        'no_html': True, 'display_winsize': 256, 'display_port': 8097,  # Display
        'print_freq': 5,  # optimizer/criterion
        'gan_mode': 'vanilla', 'optim_type': 'Adam',  # 'Adam'
        'lr': 0.002,  # train_option 中 Adam_Optimizer 的初始 learning rate, Adam 中的 default 值为 1e-3.
        'beta1': 0.9,  # train_option 中 Adam_Optimizer 的 momentum, Adam 中的 default 值为 0.9.
        # train
        'epochs': 30, # else
        'batch_size': 64, 'niter': 100, 'niter_decay': 100, 'lr_policy': 'linear', 'epoch_count': 1}