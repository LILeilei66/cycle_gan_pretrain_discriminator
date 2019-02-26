import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import os
import matplotlib.image as mpimg

# TODO: PatchNet 的 preprocessing 情况

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

TRANSFORM = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
    ])

def is_image_file(fname):
    return any(fname.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, img_list=[]):
    assert os.path.isdir(dir)

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                img_list.append(path)
    return img_list

class horseDataset(Dataset):
    """
    Fake Image -> label = 1
    Real Image -> label = 0
    """
    def __init__(self, real_dir, fake_dir, transform=TRANSFORM):
        """

        :param path_real: Path to the real imgs, e.g.: './dataset/horse/real'
        :param path_fake: Path to the fake dataset, e.g.: './dataset/horse/fake'
        :param transform:
        """
        super().__init__()
        self.real_dir = real_dir
        self.fake_dir = fake_dir
        self.transform = transform
        self.real_img_list = make_dataset(real_dir)
        self.fake_img_list = make_dataset(fake_dir)
        self.nbr_real = len(self.real_img_list)
        self.nbr_fake = len(self.fake_img_list)

    def __len__(self):
        return self.nbr_real + self.nbr_fake

    def __getitem__(self, index):
        if index >= self.nbr_real: # get fake image
            path_img = self.fake_img_list[index-self.nbr_real]
            label = 1
        else: # get real image
            path_img = self.real_img_list[index]
            label = 0

        img = mpimg.imread(path_img)
        img = self.transform(img)

        sample = {'image': img, 'label': label}

        return sample

