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
        'gpu_ids': 0,
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
        'gan_mode': 'lsgan'
        }
    for key in option_dict.keys():
        setattr(opt, key, option_dict[key])

    model = DiscriminateModel(opt)
    model.setup(opt)


