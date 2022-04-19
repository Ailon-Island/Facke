import os
import shutil
import numpy as np
import math
from options.train_options import TrainOptions
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from models.models import create_model
from data.VGGface2HQ import VGGFace2HQDataset, ComposedLoader
from utils.visualizer import Visualizer
from utils import utils
from collections import OrderedDict
import time
import tqdm
import warnings
from utils.loss import IDLoss


def lcm(a, b): return abs(a * b) / math.gcd(a, b) if a and b else 0


class Trainer:
    def __init__(self, loader, model, opt, start_epoch, epoch_iter):
        super(Trainer, self).__init__()
        self.model = model
        self.opt = opt
        self.loader = loader
        self.losses = []
        self.start_epoch = start_epoch
        self.start_epoch_iter = epoch_iter
        self.total_iter = (start_epoch - 1) * len(loader) + epoch_iter
        self.display_delta = self.total_iter % opt.display_freq
        self.print_delta = self.total_iter % opt.print_freq
        self.save_delta = self.total_iter % opt.save_latest_freq
        self.memory_last = 0
        self.memory_first = None


    def train(self, epoch_idx):
        self.model.train()

        epoch_start_time = time.time()
        epoch_iter = self.start_epoch_iter if epoch_idx == self.start_epoch else 0
        opt = self.opt

        for batch_idx, ((img_source, img_target), (latent_ID, latent_ID_target), is_same_ID) in enumerate(self.loader):
            if self.total_iter % opt.print_freq == self.print_delta:
                iter_start_time = time.time()

            batch_size = len(is_same_ID)
            self.total_iter += batch_size
            epoch_iter += batch_size

            # whether to collect output images
            save_fake = self.total_iter % opt.display_freq == self.display_delta

            is_same_ID = is_same_ID[0].detach().item()

            ########### FORWARD ###########
            if save_fake:
                [losses, img_fake] = model(img_source, img_target, latent_ID, latent_ID_target)
            else:
                [losses, _] = model(img_source, img_target, latent_ID, latent_ID_target)

            ############ LOSSES ############
            # gather losses
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]

            # loss dictionary
            loss_dict = dict(zip(model.module.loss_names, losses))

            # calculate final loss scalar
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_VGG', 0) + loss_dict.get('G_wFM', 0) + loss_dict['G_ID'] + loss_dict['G_rec'] * is_same_ID
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) + loss_dict['D_GP']

            ############ BACKWARD ############
            self.model.module.optim_G.zero_grad()
            loss_G.backward()
            self.model.module.optim_G.step()
            self.model.module.optim_D.zero_grad()
            loss_D.backward()
            self.model.module.optim_D.step()

            # save loss
            losses = [loss.detach().cpu().item() for loss in losses]
            self.losses += [losses]

            # print result
            if self.total_iter % opt.print_freq == self.print_delta:
                errors = dict(zip(model.module.loss_names, losses))
                avg_iter_time = (time.time() - iter_start_time) / opt.print_freq
                visualizer.print_current_errors(epoch_idx, epoch_iter, errors, avg_iter_time)
                visualizer.plot_current_errors(errors, self.total_iter)

            # display images
            if save_fake:
                visuals = OrderedDict([('source_img', utils.tensor2im(img_target[0])),
                                       ('id_img', utils.tensor2im(img_source[0])),
                                       ('generated_img', utils.tensor2im(img_fake.data[0]))
                                       ])
                visualizer.display_current_results(visuals, epoch_idx, self.total_iter)

           # save model
            if (self.total_iter % opt.save_latest_freq == self.save_delta):
                self.model.module.save('latest')
                self.model.module.save('{}_iter'.format(self.total_iter))
                np.savetxt(iter_path, (epoch_idx, epoch_iter), delimiter=',', fmt='%d')

            # memory log
            if opt.memory_check:
                if self.memory_first is None:
                    self.memory_first = torch.cuda.memory_allocated()
                print("Memory increase: {}MiB".format((torch.cuda.memory_allocated() - self.memory_last) / 1024. / 1024.))
                print("Total memory increase: {}MiB".format(
                    (torch.cuda.memory_allocated() - self.memory_first) / 1024. / 1024.))
                self.memory_last = torch.cuda.memory_allocated()

            # early stop
            if epoch_iter >= opt.max_dataset_size:
                break



def test(opt, model, loader, epoch_idx, total_iter):
    test_start_time = time.time()
    model.eval()

    test_iter = 0

    test_losses = []

    imgs_source = []
    imgs_target = []
    imgs_fake = []

    print('Testing...')
    for batch_idx, ((img_source, img_target), (latent_ID, latent_ID_target), is_same_ID) in enumerate(tqdm.tqdm(loader)):
        batch_size = len(is_same_ID)
        test_iter += batch_size

        is_same_ID = is_same_ID[0].detach().item()

        save_fake = test_iter % opt.display_freq_test == 0

        ########### FORWARD ###########
        if save_fake:
            [losses, img_fake] = model(img_source, img_target, latent_ID, latent_ID_target)
        else:
            [losses, _] = model(img_source, img_target, latent_ID, latent_ID_target)
        
        # gather losses
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        losses = [loss.detach().cpu().item() for loss in losses]

        # save loss
        if not len(test_losses):
            test_losses = losses
        else:
            test_losses = [
                test_loss + loss * batch_size if is_same_ID or loss_name != 'G_rec' else
                test_loss
                for loss_name, test_loss, loss in zip(model.module.loss_names, test_losses, losses)]

        # display images
        if save_fake:
            imgs_source.append(utils.tensor2im(img_target[0]))
            imgs_target.append(utils.tensor2im(img_source[0]))
            imgs_fake.append(utils.tensor2im(img_fake.data[0]))
            visuals = OrderedDict([('source_img', imgs_source),
                                   ('id_img', imgs_target),
                                   ('generated_img', imgs_fake)
                                   ])
            visualizer.display_current_results_test(visuals, epoch_idx, total_iter)
            print('\r{}-th demo testing image set for epoch {} displayed and saved.'.format(len(imgs_source), epoch_idx))

        # early stop
        if test_iter >= opt.max_dataset_size:
            break

    # print result
    test_losses = [test_loss / test_iter for test_loss in test_losses]
    test_losses = dict(zip(model.module.loss_names, test_losses))
    test_time = time.time() - test_start_time
    visualizer.print_current_errors_test(epoch_idx, epoch_iter, test_losses, test_time)
    visualizer.plot_current_errors_test(test_losses, total_iter)





if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    opt = TrainOptions().parse()

    transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((opt.image_size, opt.image_size)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    detransformer_Arcface = transforms.Compose([
        transforms.Normalize([0, 0, 0], [1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
    ])

    if opt.fp16:
        from torch.cuda.amp import autocast

    print("Generating data loaders...")
    train_data = VGGFace2HQDataset(opt, isTrain=True, transform=transformer_Arcface, is_same_ID=True, auto_same_ID=True)
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batchSize, shuffle=True, num_workers=opt.num_workers)
    test_data = VGGFace2HQDataset(opt, isTrain=False, transform=transformer_Arcface, is_same_ID=True, auto_same_ID=True)
    test_loader = DataLoader(dataset=test_data, batch_size=opt.batchSize, shuffle=False, num_workers=opt.num_workers)
    print("Dataloaders ready.")

    ###############################################################################
    # Code from
    # https://github.com/a312863063/SimSwap-train
    ###############################################################################
    save_path = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        if opt.continue_train:  # copy official checkpoint
            shutil.copyfile(os.path.join(opt.checkpoints_dir, opt.load_pretrain, 'iter.txt'),
                            os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt'))
            shutil.copyfile(os.path.join(opt.checkpoints_dir, opt.load_pretrain, 'G', 'latest.pth'),
                            os.path.join(opt.checkpoints_dir, opt.name, 'G', 'latest.pth'))
            shutil.copyfile(os.path.join(opt.checkpoints_dir, opt.load_pretrain, 'D1', 'latest.pth'),
                            os.path.join(opt.checkpoints_dir, opt.name, 'D1', 'latest.pth'))
            shutil.copyfile(os.path.join(opt.checkpoints_dir, opt.load_pretrain, 'D2', 'latest.pth'),
                            os.path.join(opt.checkpoints_dir, opt.name, 'D2', 'latest.pth'))
    if not os.path.exists(os.path.join(save_path, 'opt.txt')):
        file_name = os.path.join(save_path, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(vars(opt).items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
    else:
        start_epoch, epoch_iter = 1, 0

    opt.print_freq = lcm(opt.print_freq, opt.batchSize)
    opt.display_freq = lcm(opt.display_freq, opt.batchSize)
    opt.save_latest_freq = lcm(opt.save_latest_freq, opt.batchSize)
    opt.display_freq_test = lcm(opt.display_freq_test, opt.batchSize)
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1
        opt.display_freq_test = 1
        opt.niter = 1
        opt.niter_decay = 1
        opt.max_dataset_size = 10

    model = create_model(opt)

    trainer = Trainer(train_loader, model, opt, start_epoch, epoch_iter)

    visualizer = Visualizer(opt)

    for epoch_idx in range(start_epoch, opt.niter + opt.niter_decay + 1):
        if not opt.test_only:
            epoch_start_time = time.time()

            # train for one epoch
            if opt.fp16:
                with autocast():
                    trainer.train(epoch_idx)
            else:
                trainer.train(epoch_idx)

            epoch_time = time.time() - epoch_start_time
            print('End of epoch {}/{} \t Total time: {:.3f}'.format(epoch_idx, opt.niter + opt.niter_decay, epoch_time))

            # save model for this epoch
            if epoch_idx % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' % (epoch_idx, trainer.total_iter))
                model.module.save('latest')
                model.module.save('{}_iter'.format(trainer.total_iter))
                np.savetxt(iter_path, (epoch_idx + 1, 0), delimiter=',', fmt='%d')

            # lr decay
            if epoch_idx > opt.niter:
                model.module.update_lr()

        # test model
        test(opt, model, test_loader, epoch_idx, trainer.total_iter)




