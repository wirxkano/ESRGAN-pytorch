import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm
from constants import *

class SRGANTrainer(nn.Module):
    def __init__(self, g_network, d_network, train_loader, valid_loader, ckpt_dir=None):
        super().__init__()
        self.train_loader = train_loader
        self.train_length = len(train_loader)
        self.valid_loader = valid_loader
        self.ckpt_dir = ckpt_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Generator
        self.g_network = g_network.to(self.device)
        self.g_optimizer = torch.optim.Adam(self.g_network.parameters(), lr=G_LR, betas=(0.9, 0.999))
        self.g_scheduler = MultiStepLR(self.g_optimizer, milestones=[100, 200, 300, 400], gamma=0.5)

        # Discriminator
        self.d_network = d_network.to(self.device)
        self.d_optimizer = torch.optim.Adam(self.d_network.parameters(), lr=D_LR, betas=(0.9, 0.999))
        self.d_scheduler = MultiStepLR(self.d_optimizer, milestones=[100, 200, 300, 400], gamma=0.5)
        
        self.l1_loss = nn.L1Loss()
        self.vgg = models.vgg19(pretrained=True).features[:35].eval().to(self.device)
        for param in self.vgg.parameters():
            param.requires_grad = False
    
        self.best_psnr = 0.0

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load state_dict for generator and discriminator
        self.g_network.load_state_dict(checkpoint['generator'])
        self.d_network.load_state_dict(checkpoint['discriminator'])
        
        # Load state_dict for optimizers
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        
        # Load state_dict for schedulers
        self.g_scheduler.load_state_dict(checkpoint['g_scheduler'])
        self.d_scheduler.load_state_dict(checkpoint['d_scheduler'])
        
        start_epoch = checkpoint['epoch']
        
        return start_epoch

    def perceptual_loss(self, sr_img, hr_img):
        sr_features = self.vgg(sr_img)
        hr_features = self.vgg(hr_img)
        return nn.MSELoss()(sr_features, hr_features)

    def relativistic_gan_loss(self, real_logits, fake_logits, is_discriminator=True):
        if is_discriminator:
            real_loss = nn.BCEWithLogitsLoss()(real_logits - fake_logits.mean(), torch.ones_like(real_logits))
            fake_loss = nn.BCEWithLogitsLoss()(fake_logits - real_logits.mean(), torch.zeros_like(fake_logits))
            return (real_loss + fake_loss) / 2
        else:
            real_loss = nn.BCEWithLogitsLoss()(real_logits - fake_logits.mean(), torch.zeros_like(real_logits))
            fake_loss = nn.BCEWithLogitsLoss()(fake_logits - real_logits.mean(), torch.ones_like(fake_logits))
            return (real_loss + fake_loss) / 2

    def train(self, start_epoch=0):
        for epoch in tqdm(range(start_epoch, GAN_EPOCHS)):
            g_running_loss = 0.0
            d_running_loss = 0.0
    
            self.g_network.train()
            self.d_network.train()
            for lr_img, hr_img in self.train_loader:
                lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)

                # Discriminator update
                sr_img = self.g_network(lr_img).detach()
                real_logits = self.d_network(hr_img)
                fake_logits = self.d_network(sr_img)

                d_loss = self.relativistic_gan_loss(real_logits, fake_logits, is_discriminator=True)

                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()
                
                d_running_loss += d_loss.item()
                
                # Generator update
                sr_img = self.g_network(lr_img)
                real_logits = self.d_network(hr_img).detach()
                fake_logits = self.d_network(sr_img)

                l1_loss = self.l1_loss(sr_img, hr_img)
                perceptual_loss = self.perceptual_loss(sr_img, hr_img)
                adv_loss = self.relativistic_gan_loss(real_logits, fake_logits, is_discriminator=False)

                g_loss = perceptual_loss + LAMBDA * adv_loss + ETA * l1_loss

                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

                g_running_loss += g_loss.item()
    
            g_train_loss = g_running_loss / self.train_length
            d_train_loss = d_running_loss / self.train_length
            
            self.g_scheduler.step()
            self.d_scheduler.step()
            
            print(f'Epoch: {epoch+1}, Generator Loss: {g_train_loss}, Discriminator Loss: {d_train_loss}')
    
            if (epoch + 1) % 50 == 0:
                current_psnr, current_ssim = self.validation(self.valid_loader)
                print(f'Epoch: {epoch+1}, PSNR: {current_psnr}, SSIM: {current_ssim}')
                
                if current_psnr > self.best_psnr:
                    self.best_psnr = current_psnr
                    if not os.path.exists(self.ckpt_dir):
                        os.makedirs(self.ckpt_dir)
                    torch.save(self.g_network.state_dict(), os.path.join(self.ckpt_dir, 'esrgan_generator_best.pth'))
                
                torch.save({
                    'generator': self.g_network.state_dict(),
                    'discriminator': self.d_network.state_dict(),
                    'g_optimizer': self.g_optimizer.state_dict(),
                    'd_optimizer': self.d_optimizer.state_dict(),
                    'g_scheduler': self.g_scheduler.state_dict(),
                    'd_scheduler': self.d_scheduler.state_dict(),
                    'epoch': epoch + 1
                }, os.path.join(self.ckpt_dir, f'esrgan_checkpoint_epoch_{epoch+1}.pth'))

    def validation(self, dataloader):
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        psnr_vals = []
        ssim_vals = []
        
        self.g_network.eval()
        with torch.no_grad():
            for lr_img, hr_img in dataloader:
                lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
                output = self.g_network(lr_img)
                output = torch.clamp(output, 0.0, 1.0)
                
                psnr = psnr_metric(output, hr_img)
                ssim = ssim_metric(output, hr_img)
                psnr_vals.append(psnr.item())
                ssim_vals.append(ssim.item())
    
        mean_psnr = np.mean(psnr_vals)
        mean_ssim = np.mean(ssim_vals)
    
        psnr_metric.reset()
        ssim_metric.reset()
    
        return mean_psnr, mean_ssim

    def inference(self, lr_img):
        self.g_network.eval()
        with torch.no_grad():
            lr_img = lr_img.to(self.device)
            output = self.g_network(lr_img)
    
        return output
      