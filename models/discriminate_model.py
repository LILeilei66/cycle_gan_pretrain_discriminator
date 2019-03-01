from .base_model import BaseModel
from . import networks
from torch import optim
import itertools
from torch import load as tload

class DiscriminateModel(BaseModel):
    """
    This DescriminateModel can be used to classify inputs into two categories.
    attributions:
        netD:           本类不包含Generators.
        criterion_D:      使用LossGAN, 根本上来说是MSELoss.
        optimizer_D
        loss_names：     'D', 对于discriminator training 只需要loss_D.
        visual_names:
        model_names:
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'DescriminateModel cannot be used during training time'
        parser.set_defaults(dataset_mode='single')
        parser.add_argument('--model_suffix', type=str, default='',
                            help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')
        return parser

    def __init__(self, opt):
        """Initialize the DescriminateModel class.

        Parameters:
            opt:        存储所有需要的参量, 可由外部构建, 无需继承BaseOptions()
        """
        BaseModel.__init__(self, opt)  # specify the training losses you want to print out. The
        self.optimizers = []


        self.loss_names = ['D'] # 只需要 loss_D
        self.model_names = ['D' + opt.model_suffix]

        self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                                      opt.init_type, opt.init_gain).to(self.device)

        # assigns the model to self.netD_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netD' + opt.model_suffix, self.netD)  # store netD in self.


        if opt.optim_type == 'Adam':
            self.optimizer_D = optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        elif opt.optim_type == 'RMSprop':
            self.optimizer_D = optim.RMSprop(self.netD.parameters())
        self.optimizers.append(self.optimizer_D)
        self.criterion_D = networks.DiscriminatorLoss(opt.gan_mode).to(self.device)

    def set_input(self, input):
        """
        Parameters:
            input:      {'image': tensor array type, 'label': 0 或 1}.
        """
        self.img = input['image'].to(self.device)
        self.label = input['label']

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]

        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def  load_net(self, path):
        """
        Difference with self.load_networks():
        ------------------------------------
            1. discriminate_model 只含有一个 model, 因此，无需遍历 model_name;
            2. load_filename, 由 path 传入, 无需利用 epoch 和 name 进行 path.join;
            3. 不是很能理解 __patch_instance_norm_state_dict 在干什么.
        :param path: (str) path of the .pth state_dict file
        """
        if path is None:
            pass
        else:
            net = self.netD

            k, v = self.netD.named_parameters().__next__()
            v_origin = v.detach().mean()

            state_dict = tload(path, map_location=self.device)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            for i,key in enumerate(list(state_dict.keys())):
                self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
            net.load_state_dict(state_dict)
            k, v = self.netD.named_parameters().__next__()
            v_loaded = v.detach().mean()
            assert v_origin != v_loaded

    def forward(self):
        """Run forward pass."""
        self.features = self.netD(self.img) # D(A)

    def backward_D_basic(self, netD, image, label):
        """Calculate DiscriminatorLoss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            image (tensor array) -- images
            label (tensor array) -- {'fake': False(0), 'real': True(1)}


        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        pred = netD(image)
        loss_D = self.criterion_D(pred, label)
        loss_D.backward()
        return loss_D

    def backward_D(self):
        self.loss_D = self.backward_D_basic(self.netD, self.img, self.label)


    def optimize_parameters(self):
        """Calculate loss functions, get gradients, update network weights."""
        self.forward()      # 得到预测结果
        self.set_requires_grad(self.netD, True) # 将待优化的 netD 设为 requires_grad
        self.optimizer_D.zero_grad() # 初始化 netD 的 grad
        self.backward_D() # 计算 loss 和 grad
        self.optimizer_D.step() # 更新 netD 参数

