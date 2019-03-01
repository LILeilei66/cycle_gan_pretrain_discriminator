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
at {:}, file list length: {:}\n \n \
The number of training images = {:}\n"

NET_MESSAGE_TEMPLATE = \
"---------- Networks initialized -------------\n \
Model located at : {:}\n \
Model loaded from : {:}\n"

SAVE_MESSAGE_TEMPLATE = \
"Model save path : {:}\n"

# TODO: 为什么 load 的 net, 出来的第一个loss 会这么大？

if __name__ == '__main__':
    # TODO: 虽然 loss 于 LossGAN在减小, 但是对于<Real|Fake>的分类效果完全没有变好。
    # TODO: 三个解决方法：

    # 1. 改变 TPFN 计算方法 (最少改动)：
    #    当前: 利用 pretrained discriminator 进行 <real|fake> 分类.
    #    改动: 利用两个pretrained discriminator, 传入同一 item, 比较更接近 <horse|zebra> 中哪个.
    #    结果: 全部都会认为是zebra. (TP: 0 ; TN: 69 ; FP: 0 ; FP: 56)
    # 1.1 改变 criterion (改动亦少):
    #     当前: MSELoss (default method).
    #     改动: CrossEntropyLoss() (ZY's criterion) todo
    #     缺点: mean value 是我计算 zebra|horse 的基础, 如果改动的话, 结果就没有意义了.
    # 1.2 改变 optimizer (改动亦少):
    #     当前: Adam (default method).
    #     改动: optim.RMSprop() (之前训练 ft_vis 时的 optimizer)
    #     结果: 经过50个周期, 正确率没有提升.
    # --------------------------------------------------------------------------------------
    # 2. 改变 Discriminator 结构 (训练时间与收敛性有保证):
    #    当前: 70*70 PatchNet, 参见 'models/structure_discriminator.md'.
    #    改动: 寻找网上是否有pretrained discriminator, 比如 progressively growing gans.
    # --------------------------------------------------------------------------------------
    # 3. 改变 Loss 计算方法 (分类效果一定会变好):
    #    当前: feature([1, 30, 30]) 与 label.expanded() 的 MSELoss (论文中使用的LossGAN).
    #    改动: 增加一层, 使得new_feature([1]) 与label 进行 MSELoss.

    path_train_real_horse = './dataset/horse/train/real'
    path_train_fake_horse = './dataset/horse/train/fake'
    path_test_real_horse =  './dataset/horse/test/real'
    path_test_fake_horse =  './dataset/horse/test/fake'

    path_train_real_zebra = './dataset/zebra/train/real'
    path_train_fake_zebra = './dataset/zebra/train/fake'
    path_test_real_zebra = './dataset/zebra/test/real'
    path_test_fake_zebra = './dataset/zebra/test/fake'

    path_discriminator_horse = 'checkpoints/horse/experiment2/10_net_D.pth'

    # =============================================================================================
    # horse real|fake discriminator
    # =============================================================================================
    """
    1. 创建options; 
    2. 创建training dataset, training dataloader; 
    3. 创建clf_train_dataset, clf_train_dataloader, clf_test_dataset, clf_test_dataloader;
    4. 创建model, load model from opt.previous_model if any;
    5. Train.        
        5.1. save loss_log.txt 于 join(checkpoints_dir, name)    
             <visualizer.print_current_losses, visualizer.print_avg_loss>
        5.2. save 'test_loss.txt' % epoch 于 join(test_results_dir, name)
             <visualizer.print_test_result>        
    """
    """
    # 1. 创建options;
    opt = Option()
    option_dict = {
        'isTrain': True,
        # BaseModel.__init__()
        'gpu_ids': [0],
        'checkpoints_dir': './checkpoints/horse',
        'test_results_dir': './test_results/horse',
        'name': 'experiment5',
        'preprocess': None,
        # model.setup()
        'continue_train': False, # 虽然这里写的是False, 但是实际上则是 load net from previous_model
        'previous_model': 'checkpoints/horse/experiment4/25_net_D.pth',
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
        'display_id': -1,
        'no_html': True,
        'display_winsize': 256,
        'display_port': 8097,
        # Display
        'print_freq': 5,
        # optimizer/criterion
        'gan_mode': 'vanilla',
        'optim_type': 'Adam', # 'Adam'
        'lr': 0.002, # train_option 中 Adam_Optimizer 的初始 learning rate, Adam 中的 default 值为 1e-3.
        'beta1': 0.9, # train_option 中 Adam_Optimizer 的 momentum, Adam 中的 default 值为 0.9.
        # train
        'epochs': 30,
        # else
        'batch_size': 64,
        'niter': 100,
        'niter_decay': 100,
        'lr_policy': 'linear',
        'epoch_count': 1
        }
    for key in option_dict.keys():
        setattr(opt, key, option_dict[key])

    loss_log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    result_log_name = os.path.join(opt.test_results_dir, opt.name, 'test_loss.txt')
    print('Saving root is {:}'.format(os.path.join(opt.checkpoints_dir, opt.name)))

    # 2. 创建training dataset, training dataloader;
    train_dataset = horseDataset(real_dir=path_train_real_horse, fake_dir=path_train_fake_horse)
    dataset_size = len(train_dataset)

    data_message = DATA_MESSAGE_TEMPLATE.format( \
                    train_dataset.real_dir, len(train_dataset.real_img_list), \
                    train_dataset.fake_dir, len(train_dataset.fake_img_list), \
                    dataset_size)
    print(data_message)
    with open(loss_log_name, 'a') as log_file:
        log_file.write(data_message)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=0)

    # 3. 创建 clf_train_dataloader, clf_test_dataset, clf_test_dataloader;
    clf_train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

    clf_test_dataset = horseDataset(real_dir=path_test_real_horse, fake_dir=path_test_fake_horse)
    dataset_size = len(clf_test_dataset)
    data_message = DATA_MESSAGE_TEMPLATE.format( \
                    clf_test_dataset.real_dir, len(clf_test_dataset.real_img_list),
                    clf_test_dataset.fake_dir, len(clf_test_dataset.fake_img_list),
                    dataset_size)
    print(data_message)
    print('The number of testing images = %d' % dataset_size)
    clf_test_dataloader = DataLoader(clf_test_dataset, batch_size=1, shuffle=True, num_workers=0)

    # 4. 创建model, load model from opt.previous_model if any;
    model = DiscriminateModel(opt)
    model.setup(opt) # Load and print networks; create schedulers.
    model.load_net(opt.previous_model)
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

    net_message = NET_MESSAGE_TEMPLATE.format(model.device, opt.previous_model)
    print(net_message)
    with open(loss_log_name, 'a') as log_file:
        log_file.write(data_message)
        log_file.write(net_message)

    # 4.1 验证 Accuracy
    with open(result_log_name, "a") as test_loss_log_file:
        test_loss_log_file.write('----------Accuracy Validation -------------\n')
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    for i, data in enumerate(clf_train_dataloader):
        model.set_input(data)
        model.forward()
        prediction = model.features.mean()
        if prediction > 0.5:
            if model.label == 1:
                TP += 1
            elif model.label == 0:
                FP += 1
            else:
                raise ValueError
        else:
            if model.label == 0:
                TN += 1
            elif model.label == 1:
                FN += 1
            else:
                raise ValueError

    visualizer.print_test_result('Training', -1, TP, TN, FP, FN)

    TN = 0
    TP = 0
    FN = 0
    FP = 0
    for i, data in enumerate(clf_test_dataloader):
        model.set_input(data)
        model.forward()
        prediction = model.features.mean()
        if prediction > 0.5:
            if model.label == 1:
                TP += 1
            elif model.label == 0:
                FP += 1
            else:
                raise ValueError
        else:
            if model.label == 0:
                TN += 1
            elif model.label == 1:
                FN += 1
            else:
                raise ValueError

    visualizer.print_test_result('Testing', -1, TP, TN, FP, FN)

    # 5. Train.
    total_iters = 0                # the total number of training iterations
    for epoch in range(opt.epochs):
        running_loss = 0.0
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        epoch_loss = []

        for i, data in enumerate(train_dataloader):
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

        # 5.1. save loss_log.txt 于 join(checkpoints_dir, name)
        # <visualizer.print_current_losses, visualizer.print_avg_loss>
        visualizer.print_avg_loss(epoch, epoch_loss)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        # 5.2. save 'test_loss.txt' % epoch 于 join(test_dir, name)
        #      <visualizer.print_test_result>
        with open(result_log_name, "a") as test_loss_log_file:
            test_loss_log_file.write('---------- {:} Period Result -------------\n'.format(epoch))
        TN = 0
        TP = 0
        FN = 0
        FP = 0
        for i, data in enumerate(clf_train_dataloader):
            model.set_input(data)
            model.forward()
            prediction = model.features.mean()
            if prediction > 0.5:
                if model.label == 1:
                    TP += 1
                elif model.label == 0:
                    FP += 1
                else:
                    raise ValueError
            else:
                if model.label == 0:
                    TN += 1
                elif model.label == 1:
                    FN += 1
                else:
                    raise ValueError

        visualizer.print_test_result('Training', epoch, TP, TN, FP, FN)

        TN = 0
        TP = 0
        FN = 0
        FP = 0
        for i, data in enumerate(clf_test_dataloader):
            model.set_input(data)
            model.forward()
            prediction = model.features.mean()
            if prediction > 0.5:
                if model.label == 1:
                    TP += 1
                elif model.label == 0:
                    FP += 1
                else:
                    raise ValueError
            else:
                if model.label == 0:
                    TN += 1
                elif model.label == 1:
                    FN += 1
                else:
                    raise ValueError

        visualizer.print_test_result('Testing', epoch, TP, TN, FP, FN)
    """

    # =============================================================================================
    # zebra real|fake discriminator
    # =============================================================================================
    # 1. 创建options;
    opt = Option()
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
        'previous_model': 'checkpoints/horse/best_result/horse_net_D.pth',
        'load_iter': 0,
        'epoch': 'latest',
        'verbose': True, # DiscriminateModel.__init__()
        'model_suffix': '',
        'output_nc': 3,
        'ndf': 64,
        'netD': 'basic',
        'n_layers_D': 3,
        'norm': 'instance',
        'init_type': 'normal',
        'init_gain': 0.02,
        'display_id': -1,
        'no_html': True,
        'display_winsize': 256,
        'display_port': 8097, # Display
        'print_freq': 5, # optimizer/criterion
        'gan_mode': 'vanilla',
        'optim_type': 'Adam',  # 'Adam'
        'lr': 0.002,  # train_option 中 Adam_Optimizer 的初始 learning rate, Adam 中的 default 值为 1e-3.
        'beta1': 0.9,  # train_option 中 Adam_Optimizer 的 momentum, Adam 中的 default 值为 0.9.
        # train
        'epochs': 30,
        # else
        'batch_size': 64,
        'niter': 100,
        'niter_decay': 100,
        'lr_policy': 'linear',
        'epoch_count': 1}
    for key in option_dict.keys():
        setattr(opt, key, option_dict[key])

    loss_log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
    result_log_name = os.path.join(opt.test_results_dir, opt.name, 'test_loss.txt')
    print('Saving root is {:}'.format(os.path.join(opt.checkpoints_dir, opt.name)))

    # 2. 创建training dataset, training dataloader;
    train_dataset = horseDataset(real_dir=path_train_real_zebra, fake_dir=path_train_fake_zebra)
    dataset_size = len(train_dataset)

    data_message = DATA_MESSAGE_TEMPLATE.format( \
                train_dataset.real_dir, len(train_dataset.real_img_list), \
                train_dataset.fake_dir, len(train_dataset.fake_img_list), \
                dataset_size)
    print(data_message)
    with open(loss_log_name, 'a') as log_file:
        log_file.write(data_message)

    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=0)

    # 3. 创建 clf_train_dataloader, clf_test_dataset, clf_test_dataloader;
    clf_train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)

    clf_test_dataset = horseDataset(real_dir=path_test_real_zebra, fake_dir=path_test_fake_zebra)
    dataset_size = len(clf_test_dataset)
    data_message = DATA_MESSAGE_TEMPLATE.format( \
            clf_test_dataset.real_dir, len(clf_test_dataset.real_img_list),
            clf_test_dataset.fake_dir, len(clf_test_dataset.fake_img_list),
            dataset_size)
    print(data_message)
    print('The number of testing images = %d' % dataset_size)
    clf_test_dataloader = DataLoader(clf_test_dataset, batch_size=1, shuffle=True, num_workers=0)

    # 4. 创建model, load model from opt.previous_model if any;
    model = DiscriminateModel(opt)
    model.setup(opt) # Load and print networks; create schedulers.
    model.load_net(opt.previous_model)
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

    net_message = NET_MESSAGE_TEMPLATE.format(model.device, opt.previous_model)
    print(net_message)
    with open(loss_log_name, 'a') as log_file:
        log_file.write(data_message)
        log_file.write(net_message)

    # 4.1 验证 Accuracy
    with open(result_log_name, "a") as test_loss_log_file:
        test_loss_log_file.write('----------Accuracy Validation -------------\n')
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    for i, data in enumerate(clf_train_dataloader):
        model.set_input(data)
        model.forward()
        prediction = model.features.mean()
        if prediction > 0.5:
            if model.label == 1:
                TP += 1
            elif model.label == 0:
                FP += 1
            else:
                raise ValueError
        else:
            if model.label == 0:
                TN += 1
            elif model.label == 1:
                FN += 1
            else:
                raise ValueError

    visualizer.print_test_result('Training', -1, TP, TN, FP, FN)

    TN = 0
    TP = 0
    FN = 0
    FP = 0
    for i, data in enumerate(clf_test_dataloader):
        model.set_input(data)
        model.forward()
        prediction = model.features.mean()
        if prediction > 0.5:
            if model.label == 1:
                TP += 1
            elif model.label == 0:
                FP += 1
            else:
                raise ValueError
        else:
            if model.label == 0:
                TN += 1
            elif model.label == 1:
                FN += 1
            else:
                raise ValueError

    visualizer.print_test_result('Testing', -1, TP, TN, FP, FN)

    # 5. Train.
    total_iters = 0                # the total number of training iterations
    for epoch in range(opt.epochs):
        running_loss = 0.0
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        epoch_loss = []

        for i, data in enumerate(train_dataloader):
            iter_start_time = time.time()
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            visualizer.reset()
            model.set_input(data)       # unpack data from dataset and apply preprocessing
            model.optimize_parameters  () # calculate loss functions, get gradients,
            # update network weights

            if total_iters % opt.print_freq == 0: # print training losses and save logging
                # information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
            losses = model.get_current_losses()
            epoch_loss.append(losses['D'])

        print('saving the model at the end of epoch %d as state_dict' % (epoch))
        model.save_networks(epoch)

        # 5.1. save loss_log.txt 于 join(checkpoints_dir, name)
        # <visualizer.print_current_losses, visualizer.print_avg_loss>
        visualizer.print_avg_loss(epoch, epoch_loss)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        # 5.2. save 'test_loss.txt' % epoch 于 join(test_dir, name)
        #      <visualizer.print_test_result>
        with open(result_log_name, "a") as test_loss_log_file:
            test_loss_log_file.write('---------- {:} Period Result -------------\n'.format(epoch))
        TN = 0
        TP = 0
        FN = 0
        FP = 0
        for i, data in enumerate(clf_train_dataloader):
            model.set_input(data)
            model.forward()
            prediction = model.features.mean()
            if prediction > 0.5:
                if model.label == 1:
                    TP += 1
                elif model.label == 0:
                    FP += 1
                else:
                    raise ValueError
            else:
                if model.label == 0:
                    TN += 1
                elif model.label == 1:
                    FN += 1
                else:
                    raise ValueError

        visualizer.print_test_result('Training', epoch, TP, TN, FP, FN)

        TN = 0
        TP = 0
        FN = 0
        FP = 0
        for i, data in enumerate(clf_test_dataloader):
            model.set_input(data)
            model.forward()
            prediction = model.features.mean()
            if prediction > 0.5:
                if model.label == 1:
                    TP += 1
                elif model.label == 0:
                    FP += 1
                else:
                    raise ValueError
            else:
                if model.label == 0:
                    TN += 1
                elif model.label == 1:
                    FN += 1
                else:
                    raise ValueError

        visualizer.print_test_result('Testing', epoch, TP, TN, FP, FN)
