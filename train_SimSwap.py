import numpy as np
from options.train_options import TrainOptions
import torch
from torch import nn
from utils.IDExtract import IDExtractor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from models.models import create_model
from data.VGGface2HQ import VGGFace2HQDataset, ComposedLoader
import time
import matplotlib.pyplot as plt
import warnings
from utils.loss import IDLoss

warnings.filterwarnings("ignore")

class Trainer:
    def __init__(self, loader, model, opt):
        super(Trainer, self).__init__()

        self.model = model
        self.opt = opt
        self.loader = loader
        self.losses = []
        self.iter_cnt = 0
        self.niter_start_time = 0
        self.memory_last = 0
        self.memory_first = None


    def train(self, epoch_idx):
        for batch_idx, ((img_source, img_target), (latent_ID, latent_ID_target), is_same_ID) in enumerate(self.loader):
            self.niter_start_time = time.time()

            is_same_ID = is_same_ID[0].detach().item()

            ########### FORWARD ###########
            [losses, _] = model(img_source, img_target, latent_ID, latent_ID_target)


            ############ LOSSES ############
            loss_dict = dict(zip(model.module.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) + loss_dict['D_GP']
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_VGG', 0) + loss_dict.get('G_wFM', 0) + loss_dict['G_ID'] + \
                     loss_dict['G_rec'] * is_same_ID

            self.model.module.optim_G.zero_grad()
            loss_G.backward()
            self.model.module.optim_G.step()
            self.model.module.optim_D.zero_grad()
            loss_D.backward()
            self.model.module.optim_D.step()

            self.losses = [[loss.detach().cpu().item() for loss in losses]]

            self.iter_cnt += self.opt.batchSize

            niter_finish_time = time.time()
            niter_time = niter_finish_time - self.niter_start_time
            # self.train_half(model, img_source_diff_ID, img_target_diff_ID, is_same_ID=False)

            # display result
            if self.iter_cnt % self.opt.display_freq == 0: # two iters per batch
                self.display(self.model.module.loss_names)
                self.print(self.model.module.loss_names, epoch_idx, is_same_ID, niter_time)

            # save model
            if (self.iter_cnt % self.opt.save_latest_freq == 0):
                self.model.module.save('latest')
                # del self.model
                # torch.cuda.empty_cache()
                # self.model = create_model(self.opt)
                # self.model.module.load('latest')
                # self.model.train()
            if (self.iter_cnt % self.opt.save_epoch_freq == 0):
                self.model.module.save('{}_iter'.format(self.iter_cnt))
                # del self.model
                # torch.cuda.empty_cache()
                # self.model = create_model(self.opt)
                # self.model.module.load('{}_iter'.format(self.iter_cnt))
                # self.model.train()


            # memory log
            if opt.memory_check:
                if self.memory_first is None:
                    self.memory_first = torch.cuda.memory_allocated()
                print("Memory increase: {}MiB".format((torch.cuda.memory_allocated() - self.memory_last) / 1024. / 1024.))
                print("Total memory increase: {}MiB".format((torch.cuda.memory_allocated() - self.memory_first) / 1024. / 1024.))
                self.memory_last = torch.cuda.memory_allocated()


    def display(self, loss_names):
        # iters = range(1, self.iter_cnt + 1)
        # for i in range(img_id.shape[0]):
        #     if i == 0:
        #         img_fake_vec = img_fake[i]
        #     else:
        #         img_fake_vec = torch.cat([row3, img_fake[i]], dim=2)
        #
        # # full = torch.cat([row1, row2, row3], dim=1).detach()
        # full = row3.detach()
        # full = full.permute(1, 2, 0)
        # output = full.to('cpu')
        # output = np.array(output)
        # output = output[..., ::-1]
        #
        # output = output * 255
        #
        # cv2.imwrite(self.opt.output_path + 'result.jpg', output)
        pass


    def print(self, loss_names, epoch_idx, is_same_ID, niter_time):
        # same_ID

        # print("[epoch:\t{}\titers:\t{}:\t{}\tID:\tsame]\t\t".format(epoch_idx, self.iter_cnt - 1, niter_time), end='')
        # for (loss_name, loss) in zip(loss_names, self.losses[-2]):
        #     print("\t{}:\t{}".format(loss_name, loss), end='')
        # print()

        # diff_ID
        print("[epoch:\t{}\titers:\t{}\tsame ID:\t{}\ttime:\t{:.3f}]".format(epoch_idx, self.iter_cnt, is_same_ID, niter_time), end='')
        for (loss_name, loss) in zip(loss_names, self.losses[-1]):
            print("\t{}:\t{:.3f}".format(loss_name, loss), end='')
        print()



def test(test_loader):
    pass



if __name__ == '__main__':
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
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batchSize, shuffle=True, num_workers=1)
    # test_data = VGGFace2HQDataset(opt, isTrain=False, transform=transformer_Arcface, is_same_ID=True)
    # test_data = VGGFace2HQDataset(opt, isTrain=False, transform=transformer_Arcface, is_same_ID=False)
    # test_loader = DataLoader(dataset=test_data, batch_size=opt.batchSize, shuffle=True, num_workers=8)
    print("Dataloaders ready.")

    start_epoch, num_epochs = 1, opt.n_epochs

    model = create_model(opt)
    model.train()

    trainer = Trainer(train_loader, model, opt)

    for epoch_idx in range(1, opt.n_epochs + 1):
        if opt.fp16:
            with autocast():
                trainer.train(epoch_idx)
        else:
            trainer.train(epoch_idx)

