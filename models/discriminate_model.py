from .base_model import BaseModel
from . import networks
from torch import optim
import itertools


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
        assert(opt.isTrain)
        BaseModel.__init__(self, opt)  # specify the training losses you want to print out. The
        self.optimizers = []


        self.loss_names = ['D'] # 只需要 loss_D
        self.model_names = ['D' + opt.model_suffix]

        self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                                      opt.init_type, opt.init_gain).to(self.device)

        # assigns the model to self.netD_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        setattr(self, 'netD' + opt.model_suffix, self.netD)  # store netD in self.


        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1,
        0.999))
        self.optimizers.append(self.optimizer_D)
        self.criterion_D = networks.DiscriminatorLoss(opt.gan_mode).to(self.device)

    def set_input(self, input):
        """
        Parameters:
            input:      {'image': tensor array type, 'label': 0 或 1}.
        """
        self.img = input['image'].to(self.device)
        self.label = input['label']

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
        # self.forward()      # 得到预测结果
        self.set_requires_grad(self.netD, True) # 将待优化的 netD 设为 requires_grad
        self.optimizer_D.zero_grad() # 初始化 netD 的 grad
        self.backward_D() # 计算 loss 和 grad
        self.optimizer_D.step() # 更新 netD 参数

