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
from utils.loss import get_loss_dict
from utils.plot import plot_batch
from collections import OrderedDict
import time
import tqdm
import warnings
from utils.loss import IDLoss



detransformer_Arcface = transforms.Compose([
    transforms.Normalize([0, 0, 0], [1 / 0.229, 1 / 0.224, 1 / 0.225]),
    transforms.Normalize([-0.485, -0.456, -0.406], [1, 1, 1])
])



class Trainer:
    def __init__(self, loader, model, opt, start_epoch, epoch_iter, visualizer):
        super(Trainer, self).__init__()
        self.model              = model
        self.opt                = opt
        self.loader             = loader
        self.losses             = []
        self.start_epoch        = start_epoch
        self.start_epoch_iter   = epoch_iter
        self.total_iter         = (start_epoch - 1) * len(loader) + epoch_iter
        self.memory_last        = 0
        self.memory_first       = None
        self.visualizer         = visualizer
        self.sample_path        = os.path.join(opt.checkpoints_dir, opt.name, 'samples', 'train')
        self.sample_size        = min(8, opt.batchSize)

        if opt.verbose:
            print('Trainer initialized.')
        if opt.debug:
            print('Model instance in trainer iter: {}.'.format(self.model.module.iter))


    def train(self, epoch_idx):
        opt = self.opt

        if opt.verbose:
            print('Training...')
        if opt.debug:
            print('Model instance to be trained iter: {}.'.format(self.model.module.iter))

        epoch_start_time = time.time()
        epoch_iter      = self.start_epoch_iter if epoch_idx == self.start_epoch else 0
        visualizer      = self.visualizer
        display_delta   = self.total_iter % opt.display_freq
        print_delta     = self.total_iter % opt.print_freq
        save_delta      = self.total_iter % opt.save_latest_freq

        for batch_idx, ((img_source, img_target), (latent_ID, _), is_same_ID) in enumerate(self.loader, start=1):
            self.model.train()
            if opt.debug:
                print('Batch {}: model instance to be trained iter: {}.'.format(batch_idx, self.model.module.iter))

            if self.total_iter % opt.print_freq == print_delta:
                iter_start_time = time.time()

            if len(opt.gpu_ids):
                img_target, latent_ID = img_target.to('cuda'), latent_ID.to('cuda')


            # count iterations
            batch_size              = len(is_same_ID)
            self.total_iter        += batch_size
            self.model.module.iter  = self.total_iter
            epoch_iter             += batch_size


            is_same_ID = is_same_ID[0].detach().item()

            ########### FORWARD ###########
            [losses, _] = model(img_target, latent_ID, is_same_ID)

            ############ LOSSES ############
            # gather losses
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]

            # loss dictionary
            # loss_dict = dict(zip(self.model.module.loss_names, losses))
            loss_dict = get_loss_dict(self.model.module.loss_names, losses, opt)

            # calculate final loss scalar
            loss_G = loss_dict['G_GAN'] + loss_dict['G_Rec'] + loss_dict['G_KL'] + loss_dict['G_ID']
            loss_D = loss_dict['D_real'] + loss_dict['D_fake'] + loss_dict['D_GP']


            ############ BACKWARD ############
            self.model.module.optim.zero_grad()
            loss_G.backward()
            self.model.module.optim.step()
            self.model.module.optim_D.zero_grad()
            loss_D.backward()
            self.model.module.optim_D.step()
            
            # save loss
            losses = [loss if isinstance(loss, int) else loss.detach().cpu().item() for loss in losses]
            self.losses += [losses]

            # print result
            if self.total_iter % opt.print_freq == print_delta:
                errors = get_loss_dict(self.model.module.loss_names, losses, opt)
                avg_iter_time = (time.time() - iter_start_time) / opt.print_freq
                visualizer.print_current_errors(epoch_idx, epoch_iter, errors, avg_iter_time)
                visualizer.plot_current_errors(errors, self.total_iter)

            # display images
            if self.total_iter % opt.display_freq == display_delta:
                if not os.path.exists(self.sample_path):
                    os.mkdir(self.sample_path)
                '''
                self.model.module.M1.eval()
                self.model.module.E.eval()
                self.model.module.M2.eval()
                self.model.module.D.eval()
                '''
                self.model.module.eval()
                self.model.module.isTrain = False
                with torch.no_grad():
                    img_source = img_source[:self.sample_size].to('cuda')
                    latent_ID = latent_ID[:self.sample_size]

                    imgs = []
                    zero_img = (torch.zeros_like(img_source[0, ...]))
                    imgs.append(zero_img.cpu().numpy())
                    save_img = (detransformer_Arcface(img_source.cpu())).numpy()

                    for r in range(self.sample_size):
                        imgs.append(save_img[r, ...])

                    for i in range(self.sample_size):
                        imgs.append(save_img[i, ...])

                        image_infer = img_source[i, ...].repeat(self.sample_size, 1, 1, 1)
                        img_fake = model.module(image_infer, latent_ID).cpu().numpy()

                        for j in range(self.sample_size):
                            imgs.append(img_fake[j, ...])

                    print("Save test data for iter {}.".format(self.total_iter))
                    imgs = np.stack(imgs, axis=0).transpose(0, 2, 3, 1)
                    plot_batch(imgs, os.path.join(self.sample_path, 'step_' + str(self.total_iter) + '.jpg'))
                self.model.module.isTrain = True


                # visuals = OrderedDict([('source_img', utils.tensor2im(img_target[0])),
                #                        ('id_img', utils.tensor2im(img_source[0])),
                #                        ('generated_img', utils.tensor2im(img_fake.data[0]))
                #                        ])
                # visualizer.display_current_results(visuals, epoch_idx, self.total_iter)

           # save model
            if (self.total_iter % opt.save_latest_freq == save_delta):
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



def test(opt, model, loader, epoch_idx, total_iter, visualizer):
    test_start_time = time.time()
    model.eval()

    test_iter = 0

    test_losses = []

    print('Testing...')
    if opt.debug:
        print('Model instance being tested iter: {}.'.format(model.module.iter))
    for batch_idx, ((img_source, img_target), (latent_ID, _), is_same_ID) in enumerate(tqdm.tqdm(loader)):
        batch_size = len(is_same_ID)
        test_iter += batch_size

        if len(opt.gpu_ids):
            img_target, latent_ID = img_target.to('cuda'), latent_ID.to('cuda')

        is_same_ID = is_same_ID[0].detach().item()

        ########### FORWARD ###########
        [losses, _] = model(img_target, latent_ID, is_same_ID)
        
        # gather losses
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        losses = [loss.detach().cpu().item() for loss in losses]

        # save loss
        if not len(test_losses):
            test_losses = losses
        else:
            test_losses = [
                test_loss + loss * batch_size
                for test_loss, loss in zip(test_losses, losses)]

        # display images
        if batch_idx == 0:
            sample_size = min(8, opt.batchSize)
            sample_path = os.path.join(opt.checkpoints_dir, opt.name, 'samples', 'test')

            if not os.path.exists(sample_path):
                os.mkdir(sample_path)
            '''
            self.model.module.M1.eval()
            self.model.module.E.eval()
            self.model.module.M2.eval()
            self.model.module.D.eval()
            '''
            model.module.eval()
            model.module.isTrain = False
            with torch.no_grad():
                img_source = img_source[:sample_size].to('cuda')
                latent_ID = latent_ID[:sample_size]

                imgs = []
                zero_img = (torch.zeros_like(img_source[0, ...]))
                imgs.append(zero_img.cpu().numpy())
                save_img = (detransformer_Arcface(img_source.cpu())).numpy()

                for r in range(sample_size):
                    imgs.append(save_img[r, ...])

                for i in range(sample_size):
                    imgs.append(save_img[i, ...])

                    image_infer = img_source[i, ...].repeat(sample_size, 1, 1, 1)
                    img_fake = model.module(image_infer, latent_ID).cpu().numpy()

                    for j in range(sample_size):
                        imgs.append(img_fake[j, ...])

                imgs = np.stack(imgs, axis=0).transpose(0, 2, 3, 1)
                plot_batch(imgs, os.path.join(sample_path, 'step_' + str(total_iter) + '.jpg'))
            model.module.isTrain = True

        # early stop
        if test_iter >= opt.max_dataset_size:
            break

    # print result
    test_losses = [test_loss / test_iter for test_loss in test_losses]
    test_losses = get_loss_dict(model.module.loss_names, test_losses, opt)
    test_time = time.time() - test_start_time
    visualizer.print_current_errors_test(epoch_idx, total_iter, test_losses, test_time)
    visualizer.plot_current_errors_test(test_losses, total_iter)





if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.benchmark = True
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    opt = TrainOptions().parse()

    if len(opt.gpu_ids):
        print('GPU available: {}'.format(torch.cuda.is_available()))
        print('GPU count: {}'.format(torch.cuda.device_count()))

    transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((opt.image_size, opt.image_size)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if opt.fp16:
        from torch.cuda.amp import autocast

    print("Generating data loaders...")
    train_data = VGGFace2HQDataset(opt, isTrain=True, transform=transformer_Arcface, is_same_ID=True, auto_same_ID=True)
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batchSize, shuffle=True, num_workers=opt.nThreads, worker_init_fn=train_data.set_worker)
    test_data = VGGFace2HQDataset(opt, isTrain=False, transform=transformer_Arcface, is_same_ID=True, auto_same_ID=True)
    test_loader = DataLoader(dataset=test_data, batch_size=opt.batchSize, shuffle=True, num_workers=opt.nThreads, worker_init_fn=train_data.set_worker)
    print("Dataloaders ready.")
    opt.max_dataset_size = min(opt.max_dataset_size, len(train_data))

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
            shutil.copyfile(os.path.join(opt.checkpoints_dir, opt.load_pretrain, 'M1', 'latest.pth'),
                            os.path.join(opt.checkpoints_dir, opt.name, 'M1', 'latest.pth'))
            shutil.copyfile(os.path.join(opt.checkpoints_dir, opt.load_pretrain, 'E', 'latest.pth'),
                            os.path.join(opt.checkpoints_dir, opt.name, 'E', 'latest.pth'))
            shutil.copyfile(os.path.join(opt.checkpoints_dir, opt.load_pretrain, 'M2', 'latest.pth'),
                            os.path.join(opt.checkpoints_dir, opt.name, 'M2', 'latest.pth'))
            shutil.copyfile(os.path.join(opt.checkpoints_dir, opt.load_pretrain, 'D', 'latest.pth'),
                            os.path.join(opt.checkpoints_dir, opt.name, 'D', 'latest.pth'))                            
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

    sample_path = os.path.join(opt.checkpoints_dir, opt.name, 'samples')
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)

    model = create_model(opt)
    if opt.debug:
        print('Model instance iter: {}.'.format(model.module.iter))

    visualizer = Visualizer(opt)

    trainer = Trainer(train_loader, model, opt, start_epoch, epoch_iter, visualizer)

    for epoch_idx in range(start_epoch, opt.niter + opt.niter_decay + 1):
        if opt.isTrain:
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
            if epoch_idx >= opt.niter:
                model.module.update_lr()
                if opt.verbose:
                    print('Learning rate has been changed to {}.'.format(model.module.old_lr))


        # test model
        test(opt, model, test_loader, epoch_idx, trainer.total_iter, visualizer)




