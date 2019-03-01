"""
利用已经经过训练的 horse_net_D, zebra_net_D 进行 horse|zebra classification.
"""
from models.discriminate_model import DiscriminateModel
from util.visualizer import Visualizer
from data.discriminator_dataset import horseDataset
from torch.utils.data import DataLoader
import os

"""

 ================ Testing Loss (Fri Mar  1 15:46:12 2019) ================
 confuse : 0
 -1 classification :
 TP = 1004 ; TN = 806 ; FP = 325 ; FN = 401 
 Accuracy = 0.7137223974763407 ; Recall = 0.7554552294958615 ; Precision = 0.7145907473309608 
 
 问题及改进方式：
 


"""
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

if __name__ == '__main__':
    path_train_real_horse = './dataset/horse/train/real'
    path_train_fake_horse = './dataset/horse/train/fake'
    path_test_real_horse =  './dataset/horse/test/real'
    path_test_fake_horse =  './dataset/horse/test/fake'

    path_train_real_zebra = './dataset/zebra/train/real'
    path_test_real_zebra = './dataset/zebra/test/real'

    path_model_horse = './checkpoints/horse/best_result/horse_net_D.pth'
    path_model_zebra = './checkpoints/zebra/best_result/zebra_net_D.pth'

    # 1. 创建 options
    opt = Option()
    option_dict = {
        'isTrain': True,
        # BaseModel.__init__()
        'gpu_ids': [0],
        'checkpoints_dir': './checkpoints/classification',
        'test_results_dir': './test_results/classification',
        'name': 'experiment1',
        'preprocess': None,
        # model.setup()
        'continue_train': False,  # 虽然这里写的是False, 但是实际上则是 load net from previous_model
        'previous_model': None, # Pass
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
        # optimizer/criterion
        'gan_mode': 'vanilla',
        'optim_type': 'Adam',  # 'Adam'
        'lr': 0.002,  # train_option 中 Adam_Optimizer 的初始 learning rate, Adam 中的 default 值为 1e-3.
        'beta1': 0.9,  # train_option 中 Adam_Optimizer 的 momentum, Adam 中的 default 值为 0.9.
        # else
        'batch_size': 64,
        'niter': 100,
        'niter_decay': 100,
        'lr_policy': 'linear',
        'epoch_count': 1
        }

    for key in option_dict.keys():
        setattr(opt, key, option_dict[key])

    # 2. 创建 datasets
    dataset = horseDataset(real_dir=path_train_real_zebra, fake_dir=path_train_real_horse)
    horse_dataloader = DataLoader(dataset)
    data_message = DATA_MESSAGE_TEMPLATE.format(
        dataset.real_dir, len(dataset.real_img_list),
        dataset.fake_dir, len(dataset.fake_img_list),
        len(dataset))
    print(data_message)


    # 3. 创建 models
    horse_model = DiscriminateModel(opt)
    horse_model.load_net(path_model_horse)
    net_message = NET_MESSAGE_TEMPLATE.format(horse_model.device, path_model_horse)
    print(net_message)
    zebra_model = DiscriminateModel(opt)
    zebra_model.load_net(path_model_zebra)
    net_message = NET_MESSAGE_TEMPLATE.format(zebra_model.device, path_model_zebra)
    print(net_message)

    # 4. 创建 visualizer
    visualizer = Visualizer(opt)

    # 5. 进行 classification
    """
    记 zebra 为 P, horse 为 N.
    """
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    confuse = 0
    for i, data in enumerate(horse_dataloader):
        horse_model.set_input(data)
        horse_model.forward()
        horse_prediction = horse_model.features.detach().mean()
        zebra_model.set_input(data)
        zebra_model.forward()
        zebra_prediction = zebra_model.features.detach().mean()

        if horse_prediction > zebra_prediction:
            if data['label'] == 1:
                FN += 1
            elif data['label'] == 0:
                TN += 1
        elif zebra_prediction > horse_prediction:
            if data['label'] == 1:
                TP += 1
            elif data['label'] == 0:
                FP += 1
        else:
            confuse += 1
    print(confuse)
    visualizer.print_test_result(period='classification', epoch=-1, TP=TP, TN=TN, FP=FP, FN=FN)
