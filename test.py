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
"Log save path : {:}\n"

if __name__ == '__main__':
    path_real_horse = './dataset/horse/real'
    path_real_zebra = './dataset/zebra/real'
    path_fake_horse = './dataset/horse/fake'
    path_fake_zebra = './dataset/zebra/fake'

    path_discriminator_horse = './checkpoints/horse/experiment2/20_net_D.pth' # Average loss at 20 th epoch is 0.253

    path_discriminator_zebra = './checkpoints/zebra/experiment1/12_net_D.pth' # Average loss at 12 th epoch is 0.253


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
        'batch_size': 128,
        'niter': 100,
        'niter_decay': 100,
        'lr_policy': 'linear',
        'epoch_count': 1
        }

    opt = Option()
    for key in option_dict.keys():
        setattr(opt, key, option_dict[key])

    # 2. Generate data;
    dataset = horseDataset(path_real_horse, path_real_zebra)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

    dataset_size = len(dataset)

    data_message = DATA_MESSAGE_TEMPLATE.format( \
                    dataset.real_dir, len(dataset.real_img_list), \
                    dataset.fake_dir, len(dataset.fake_img_list))
    print(data_message)
    print('The number of training images = %d' % dataset_size)

    # 3. Create and load models;
    model = DiscriminateModel(opt)
    model.load_net(path_discriminator_horse)
    model.print_networks(opt.verbose)

    # 4. Test;
    # TODO
    """
    1) 传入 1 batch 的 image; 
    2) Compare NET result with labels.
    
    当进行训练的时候, 是让最后的结果往 <0|1> 靠近的，所以直接拿 features.mean() 与 0.5 比较, 作为一个比较简单的
    """

    sample_real = dataset.__getitem__(0)
    sample_real['image'] = tunsqueeze(sample_real['image'], 0) # 为 img 增加一个维度, 当前 shape 为 [1,3,256,256]
    label = sample_real['label']

    model.set_input(sample_real)
    model.forward()
    print(model.features.shape)
    print(model.features.mean())

    sample_fake = dataset.__getitem__(0)
    sample_real['image'] = tunsqueeze(sample_real['image'], 0) # 为 img 增加一个维度, 当前 shape 为 [1,3,256,256]
    label = sample_real['label']

    model.set_input(sample_real)
    model.forward()
    print(model.features.shape)
    print(model.features.mean())






