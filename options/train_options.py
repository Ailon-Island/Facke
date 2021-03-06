###############################################################################
# Code from
# https://github.com/neuralchen/SimSwap
###############################################################################
from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=10000, help='frequency of showing training results on screen')
        self.parser.add_argument('--display_freq_test', type=int, default=10000, help='frequency of showing testing results on screen')
        self.parser.add_argument('--print_freq', type=int, default=800, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=10000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')

        # for training
        self.parser.add_argument("--memory_check", action='store_true', help='check for unexpected memory increase batch by batch')
        self.parser.add_argument("--ID_check", action='store_true', help='check if the dataloader generates images with correct ID')
        self.parser.add_argument("--Arc_path", type=str, default='utils/Arcface/arcface_checkpoint.tar', help="run ONNX model via TRT")
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='', help='load the pretrained model from the specified location')
        self.parser.add_argument('--epoch_label', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--niter', type=int, default=2, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0004, help='initial learning rate for adam')

        # for discriminators        
        self.parser.add_argument('--num_D', type=int, default=2, help='number of discriminators to use')
        self.parser.add_argument('--n_layers_D', type=int, default=4, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')    
        self.parser.add_argument('--lambda_FM', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument('--lambda_id', type=float, default=30.0, help='weight for id loss')
        self.parser.add_argument('--lambda_rec', type=float, default=10.0, help='weight for reconstruction loss')
        self.parser.add_argument('--lambda_rec_swap', type=float, default=8.0, help='weight for reconstruction loss in CVAE while swapping')
        self.parser.add_argument('--lambda_GP', type=float, default=1E-5, help='weight for gradient penalty loss')
        self.parser.add_argument('--lambda_KL', type=float, default=0.00025, help='weight for KL loss in CVAE')
        self.parser.add_argument('--lambda_KL_out', type=float, default=10.0, help='outside weight for KL loss in CVAE-GAN')
        self.parser.add_argument('--feat_mode', type=str, default='w', help='the variant of feature matching to use, default weak')
        self.parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')        
        self.parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        self.parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--times_G', type=int, default=1,
                                 help='time of training generator before traning discriminator')

        # for ILVR
        self.parser.add_argument('--ema_rate', type=str, default="0.9999")
        self.parser.add_argument('--schedule_sampler', type=str, default="uniform")
        self.parser.add_argument('--weight_decay', type=float, default=0.0)


        self.isTrain = True
