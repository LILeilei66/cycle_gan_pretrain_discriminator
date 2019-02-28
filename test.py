"""
Test 2 pretrained discriminators.
1. Classification accuracy for <fake|real> images.TODO
2. Classification accuracy for <horse|zebra> images. TODO
"""
from data.discriminator_dataset import horseDataset
from models.discriminate_model import DiscriminateModel
from torch.utils.data import DataLoader
from torch import unsqueeze as tunsqueeze
import os
import time

class Option():
    pass

DATA_MESSAGE_TEMPLATE = \
"---------- Dataset initialized -------------\n \
at {:}, file list length: {:}\n \
at {:}, file list length: {:}\n"

NET_MESSAGE_TEMPLATE = \
"---------- Networks initialized -------------\n \
Load network from {:}\n"

SAVE_MESSAGE_TEMPLATE = \
"Log save path : {:}\n"

CLF_MESSAGE_TEMPLATE = \
"---------- Classification result -------------\n \
TP = {:} ; TN = {:} ; FP = {:} ; FN = {:} \n \
Accuracy = {:} \n \
Recall = {:} \n \
Precision = {:} \n \
"

if __name__ == '__main__':
    path_real_horse = './dataset/horse/real'
    path_real_zebra = './dataset/zebra/real'
    path_fake_horse = './dataset/horse/fake'
    path_fake_zebra = './dataset/zebra/fake'

    pth_numerate_horse = 29
    path_discriminator_horse = './checkpoints/horse/experiment1/%d_net_D.pth' % pth_numerate_horse
    # Average loss at 20 th epoch is 0.253
    pth_numerate_zebra = 12
    path_discriminator_zebra = './checkpoints/zebra/experiment1/%d_net_D.pth' % pth_numerate_zebra
    # Average loss at 12 th epoch is 0.253


    # =============================================================================================
    # Test horse real|fake discriminator
    # =============================================================================================
    """
    1. Create options;
    2. Generate data;
    3. Create and load models;
    4. Test; 
    5. Save results.
    """
    print('=====Test horse real|fake discriminator=========')
    # 1. Create options;
    option_dict = {
        'isTrain': False,
        # BaseModel.__init__()
        'gpu_ids': [0],
        'test_results_dir': './test_results/horse',
        'checkpoints_dir': './test_results/horse',
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
        'batch_size': 1, # 在测试中，使用batch_size 为 1
        'niter': 100,
        'niter_decay': 100,
        'lr_policy': 'linear',
        'epoch_count': 1
        }

    opt = Option()
    for key in option_dict.keys():
        setattr(opt, key, option_dict[key])

    log_path = os.path.join(opt.test_results_dir, opt.name, '%d_net_D.txt' % pth_numerate_horse)
    assert os.path.isfile(log_path)
    with open(log_path, 'a') as log_file:
        now = time.strftime('%c')
        log_file.write('================ Test Result (%s) ================\n' % now)

    # 2. Generate data;
    dataset = horseDataset(path_real_horse, path_real_zebra)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

    dataset_size = len(dataset)

    data_message = DATA_MESSAGE_TEMPLATE.format( \
                    dataset.real_dir, len(dataset.real_img_list), \
                    dataset.fake_dir, len(dataset.fake_img_list))
    print(data_message)
    print('The number of training images = %d' % dataset_size)
    with open(log_path, 'a') as log_file:
        log_file.write(data_message)
        log_file.write('The number of training images = %d\n' % dataset_size)

    # 3. Create and load models;
    model = DiscriminateModel(opt)
    model.load_net(path_discriminator_horse)
    model.print_networks(opt.verbose)
    net_message = NET_MESSAGE_TEMPLATE.format(path_discriminator_horse)
    with open(log_path, 'a') as log_file:
        log_file.write(net_message)

    # 4. Test;
    """
    1) 传入 1 batch 的 image; 
    2) Compare NET result with labels.
    
    当进行训练的时候, 是让最后的结果往 <0|1> 靠近的，所以直接拿 features.mean() 与 0.5 比较, 作为一个比较简单的
    """
    TN = 0
    TP = 0
    FN = 0
    FP = 0

    for i, data in enumerate(dataloader):
        model.set_input(data)
        model.forward()
        prediction = model.features.mean()
        if model.label == 1:
            if prediction > 0.5:
                TP += 1
            else:
                FN += 1
        elif model.label == 0:
            if prediction > 0.5:
                FP += 1
            else:
                TN += 1
    # print(TP, FN, TN, FP)

    P = TP / (TP + FP) # Precision 查准率
    R = TP / (TP + FN) # Recall 查全率
    Acc = (TP + TN) / (TP + TN + FP + FN)

    clf_message = CLF_MESSAGE_TEMPLATE.format(TP, TN, FP, FN, Acc, P,R)
    print(clf_message)
    with open(log_path, 'a') as log_file:
        log_file.write(clf_message)




