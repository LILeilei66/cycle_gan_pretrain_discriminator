"""
Pretrain discriminator:
1. generate fake images (DONE)
2. pretrain
"""

# TODO: create dataloader < Real | Fake >

import time
from data import create_dataset
from models.discriminate_model import DiscriminateModel
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data.discriminator_dataset import horseDataset
from torch.utils.data import DataLoader
import sys
import os
from util.visualizer import Visualizer
from models.networks import GANLoss
from torch import optim
import time

class Option():
    pass

if __name__ == '__main__':
    path_real_horse = './dataset/horse/real'
    path_real_zebra = './dataset/zebra/real'
    path_fake_horse = './dataset/horse/fake'
    path_fake_zebra = './dataset/zebra/fake'

    # =============================================================================================
    # horse real|fake discriminator
    # =============================================================================================


    dataset = horseDataset(real_dir=path_real_horse, fake_dir=path_fake_horse)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

    opt = Option()
    option_dict = {
        'isTrain': True,
        # BaseModel.__init__()
        'gpu_ids': [0],
        'checkpoints_dir': './checkpoints',
        'name': 'experiment1',
        'preprocess': None,
        # model.setup()
        'continue_train': False,
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
        # else
        'batch_size': 128
        }
    for key in option_dict.keys():
        setattr(opt, key, option_dict[key])

    model = DiscriminateModel(opt)
    model.setup(opt)

    # item = dataset.__getitem__(0)
    # print(item['image'].shape)
    # print(model.netD(dataset.__getitem__(0)['image']))

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    # criterion = GANLoss('lsgan')
    # optimizer = optim.RMSprop(model.parameters())


    total_iters = 0                # the total number of training iterations
    # TODO: 准备训练
    for epoch in range(20):
        running_loss = 0.0
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(dataloader):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            visualizer.reset()
            model.set_input(data)       # unpack data from dataset and apply preprocessing
            model.optimize_parameters() # calculate loss functions, get gradients, update network weights

            if total_iters % opt.print_freq == 0: # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)