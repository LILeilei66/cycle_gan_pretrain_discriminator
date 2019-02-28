"""
Pretrain discriminator:
1. generate fake images (DONE)
2. pretrain (DONE)
"""

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

DATA_MESSAGE_TEMPLATE = \
"---------- Dataset initialized -------------\n \
at {:}, file list length: {:}\n \
at {:}, file list length: {:}\n"

NET_MESSAGE_TEMPLATE = \
"---------- Networks initialized -------------\n \
Model located at : {:}\n"

SAVE_MESSAGE_TEMPLATE = \
"Model save path : {:}\n"

if __name__ == '__main__':
    path_real_horse = './dataset/horse/real'
    path_real_zebra = './dataset/zebra/real'
    path_fake_horse = './dataset/horse/fake'
    path_fake_zebra = './dataset/zebra/fake'

    # =============================================================================================
    # horse real|fake discriminator
    # =============================================================================================

    opt = Option()
    option_dict = {
        'isTrain': True,
        # BaseModel.__init__()
        'gpu_ids': [0],
        'checkpoints_dir': './checkpoints/horse',
        'name': 'experiment2',
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
        # Train
        'lr': 0.002, # train_option 中 Adam_Optimizer 的初始 learning rate, Adam 中的 default 值为 1e-3.
        'beta1': 0.5, # train_option 中 Adam_Optimizer 的 momentum, Adam 中的 default 值为 0.9.
        'epochs': 30,
        # else
        'batch_size': 256,
        'niter': 100,
        'niter_decay': 100,
        'lr_policy': 'linear',
        'epoch_count': 1
        }
    for key in option_dict.keys():
        setattr(opt, key, option_dict[key])

    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

    dataset = horseDataset(real_dir=path_real_horse, fake_dir=path_fake_horse)
    dataset_size = len(dataset)

    data_message = DATA_MESSAGE_TEMPLATE.format( \
                    dataset.real_dir, len(dataset.real_img_list), \
                    dataset.fake_dir, len(dataset.fake_img_list))
    print(data_message)
    print('The number of training images = %d' % dataset_size)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0)

    model = DiscriminateModel(opt)
    net_message = NET_MESSAGE_TEMPLATE.format(model.device)
    print(net_message)
    model.setup(opt) # Load and print networks; create schedulers.
    path_discriminator_horse = 'checkpoints/horse/experiment1/29_net_D.pth'
    model.load_net(path_discriminator_horse)
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

    with open(log_name, 'a') as log_file:
        log_file.write(data_message)
        log_file.write(net_message)

    total_iters = 0                # the total number of training iterations
    for epoch in range(opt.epochs):
        running_loss = 0.0
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        epoch_loss = []
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
            losses = model.get_current_losses()
            epoch_loss.append(losses['D'])

        print('saving the model at the end of epoch %d as state_dict' % (epoch))
        model.save_networks(epoch)

        visualizer.print_avg_loss(epoch, epoch_loss)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))


    # =============================================================================================
    # zebra real|fake discriminator
    # =============================================================================================
    """
    opt = Option()
    option_dict = {
        'isTrain': True,
        # BaseModel.__init__()
        'gpu_ids': [0],
        'checkpoints_dir': './checkpoints/zebra',
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
        # Train
        'lr': 0.002, # train_option 中 Adam_Optimizer 的初始 learning rate, Adam 中的 default 值为 1e-3.
        'beta1': 0.5, # train_option 中 Adam_Optimizer 的 momentum, Adam 中的 default 值为 0.9.
        'epochs': 30,
        # else
        'batch_size': 128,
        'niter': 100,
        'niter_decay': 100,
        'lr_policy': 'linear',
        'epoch_count': 1
        }
    for key in option_dict.keys():
        setattr(opt, key, option_dict[key])

    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

    save_message = SAVE_MESSAGE_TEMPLATE.format(os.path.join(opt.checkpoints_dir, opt.name))
    print(save_message)

    dataset = horseDataset(real_dir=path_real_zebra, fake_dir=path_fake_zebra)
    dataset_size = len(dataset)

    data_message = DATA_MESSAGE_TEMPLATE.format( \
                    dataset.real_dir, len(dataset.real_img_list), \
                    dataset.fake_dir, len(dataset.fake_img_list))
    print(data_message)
    print('The number of training images = %d' % dataset_size)

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

    model = DiscriminateModel(opt)
    net_message = NET_MESSAGE_TEMPLATE.format(model.device)
    print(net_message)
    model.setup(opt) # Load and print networks; create schedulers.
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

    with open(log_name, 'a') as log_file:
        log_file.write(save_message)
        log_file.write(data_message)
        log_file.write(net_message)

    total_iters = 0                # the total number of training iterations
    for epoch in range(opt.epochs):
        running_loss = 0.0
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        epoch_loss = []
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
            losses = model.get_current_losses()
            epoch_loss.append(losses['D'])

        print('saving the model at the end of epoch %d as state_dict' % (epoch))
        model.save_networks(epoch)

        visualizer.print_avg_loss(epoch, epoch_loss)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    """