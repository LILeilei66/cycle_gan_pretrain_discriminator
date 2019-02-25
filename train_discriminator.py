"""
Discriminators composition:
===========================
    70*70 PatchNet + ResNet18(待定)
    1. 70*70 PatchNet
        论文中所使用的discriminator模型。
        特点：
        -----
        InstanceNorm
    2. ReNet18(待定)
        优点:
        -----
        收敛性比较有保证；使用B atchNorm 比起 InstanceNorm 更加适合分类问题。

        缺点：
        -----
        也许能够找到更加新的，正确率更加高的Net。

discriminator training strategy:
================================
    1. Pre-train 70*70 PatchNet:
    ----------------------------
    利用 horse2zebra.pth 生成 fake_zebra 图片；
    训练 PatchNet 进行 discrimination。

    利用 zebra2horse.pth 生成 fake_horse 图片；
    start with horse2zebra_discriminator 参数(待定)。
    训练 PatchNet进行 discrimination。

    2. train ResNet18 作为第二个 discriminator, 两个 discriminator 结合进行 clf, 比较分类正确率。
"""

import time
# import corresponding options
from models import create_model
from util.visualizer import Visualizer
from data.discriminator_dataset import horseDataset

if __name__ == '__main__':

    dataset = horseDataset()

