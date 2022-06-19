###############################################################################
# Code from
# https://github.com/neuralchen/SimSwap
###############################################################################
'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-23 17:03:58
LastEditors: Naiyuan liu
LastEditTime: 2021-11-23 17:08:08
Description: 
'''
from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float("100"), help='# of test examples.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--epoch_label', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')       
        self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
        self.parser.add_argument('--use_encoded_image', action='store_true', help='if specified, encode the real image to get the feature map')
        self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
        self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")        
        self.parser.add_argument("--Arc_path", type=str, default='utils/Arcface/arcface_checkpoint.tar', help="run ONNX model via TRT")
        self.parser.add_argument("--Cos_path", type=str, default='utils/Cosface/cosface_checkpoint.pth', help="CosFace model checkpoint")
        self.parser.add_argument("--pic_a_path", type=str, default='./crop_224/gdg.jpg', help="Person who provides identity information")
        self.parser.add_argument("--pic_b_path", type=str, default='./crop_224/zrf.jpg', help="Person who provides information other than their identity")
        self.parser.add_argument("--pic_specific_path", type=str, default='./crop_224/zrf.jpg', help="The specific person to be swapped")
        self.parser.add_argument("--multisepcific_dir", type=str, default='./demo_file/multispecific', help="Dir for multi specific")
        self.parser.add_argument("--video_path", type=str, default='./demo_file/multi_people_1080p.mp4', help="path for the video to swap")
        self.parser.add_argument("--temp_path", type=str, default='./temp_results', help="path to save temporarily images")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="results path")
        self.parser.add_argument("--source_path", type=str, default='./input/source', help="source image path")
        self.parser.add_argument("--target_path", type=str, default='./input/target', help="target image path")
        self.parser.add_argument('--id_thres', type=float, default=0.03, help='how many test images to run')
        self.parser.add_argument('--no_simswaplogo', action='store_true', help='Remove the watermark')
        self.parser.add_argument('--use_mask', action='store_true', help='Use mask for better result')
        self.parser.add_argument('--crop_size', type=int, default=224, help='Crop of size of input image')
        self.parser.add_argument('--finetuned', action='store_true')
        self.parser.add_argument('--benchmark_coarse', type=int, default=4000, help='samples to run in coarse benchmark')
        self.parser.add_argument('--benchmark_fine', type=int, default=20000, help='samples to run in fine benchmark')
        self.parser.add_argument('--benchmark_skip', type=int, default=1, help='samples to run in fine benchmark')
        self.parser.add_argument('--num_to_swap', type=int, default=8, help='# of images to get swapped')



        self.isTrain = False
