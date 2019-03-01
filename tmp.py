# 为什么save, load 结果不同?

from models.discriminate_model import DiscriminateModel
from data.discriminator_dataset import horseDataset
from torch.utils.data import DataLoader
import os
from util.visualizer import Visualizer

class Option():
    pass

DATA_MESSAGE_TEMPLATE = \
"---------- Dataset initialized -------------\n \
at {:}, file list length: {:}\n \
at {:}, file list length: {:}\n \n \
The number of training images = {:}\n"

NET_MESSAGE_TEMPLATE = \
"---------- Networks initialized -------------\n \
Model located at : {:}\n \
Model loaded from : {:}\n"

SAVE_MESSAGE_TEMPLATE = \
"Model save path : {:}\n"

if __name__ == '__main__':
    path_train_real_horse = './dataset/horse/train/real'
    path_train_fake_horse = './dataset/horse/train/fake'

    # 1. 创建 options
    opt = Option()

    option_dict = {
        'isTrain': True,
        # BaseModel.__init__()
        'gpu_ids': [0],
        'checkpoints_dir': './checkpoints/horse',
        'test_results_dir': './test_results/horse',
        'name': 'experiment3',
        'preprocess': None,
        # model.setup()
        'continue_train': False, # 虽然这里写的是False, 但是实际上则是 load net from previous_model
        'previous_model': None,
        'load_iter': 0,
        'epoch': 'latest',
        'verbose': True,
        # DiscriminateModel.__init__()
        'model_suffix': '',
        'output_nc': 3,
        'ndf': 64,
        'netD': 'basic',
        'n_layers_D': 3,
        'norm': 'instance',
        'init_type': 'normal',
        'init_gain': 0.02,
        'gan_mode': 'lsgan',
        'display_id': -1,
        'no_html': True,
        'display_winsize': 256,
        'display_port': 8097,
        # Display
        'print_freq': 5,
        # optimizer
        'optim_type': 'Adam', # 'Adam'
        'lr': 0.002, # train_option 中 Adam_Optimizer 的初始 learning rate, Adam 中的 default 值为 1e-3.
        'beta1': 0.5, # train_option 中 Adam_Optimizer 的 momentum, Adam 中的 default 值为 0.9.
        # train
        'epochs': 1,
        # else
        'batch_size': 64,
        'niter': 100,
        'niter_decay': 100,
        'lr_policy': 'linear',
        'epoch_count': 1
        }
    for key in option_dict.keys():
        setattr(opt, key, option_dict[key])

    # 2. 创建 model
    model = DiscriminateModel(opt)
    model.setup(opt)

    # 3. 创建 dataset
    dataset = horseDataset(real_dir=path_train_real_horse, fake_dir=path_train_fake_horse)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

    # 4. 创建visualizer
    visualizer = Visualizer(opt)

    # 5. 开始训练
    for epoch in range(opt.epochs):
        epoch_iter = 0
        for i, data in enumerate(dataloader):
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            losses = model.get_current_losses()
            visualizer.print_current_losses(epoch, epoch_iter, losses, -1, -1)
    model.save_networks(epoch)

    # 6. 检查 save
    print('========Reload saved nets======================')
    model2 = DiscriminateModel(opt)
    model2.setup(opt)
    visualizer = Visualizer(opt)
    epoch_iter = 0
    for i, data in enumerate(dataloader):
        epoch_iter += opt.batch_size
        model.set_input(data)
        model.optimize_parameters()
        losses = model.get_current_losses()
        visualizer.print_current_losses(epoch, epoch_iter, losses, -1, -1)