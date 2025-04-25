import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm
from constants import *

class SRTrainer(nn.Module):
    def __init__(self, network, train_loader, valid_loader, ckpt_dir=None):
        super().__init__()
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_length = len(train_loader)
        self.ckpt_dir = ckpt_dir
    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = network.to(self.device)
    
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.5)
        self.loss_fn = nn.L1Loss()
    
        self.best_psnr = 0.0

    def train(self):
        for epoch in tqdm(range(EPOCHS)):
            running_loss = 0.0
    
            self.network.train()
            for lr_img, hr_img in self.train_loader:
                lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
        
                self.optimizer.zero_grad()
                output = self.network(lr_img)
                loss = self.loss_fn(output, hr_img)
                loss.backward()
                self.optimizer.step()
        
                running_loss += loss.item()
    
            train_loss = running_loss / self.train_length
            self.scheduler.step()
            print(f'Epoch: {epoch+1}, Train Loss: {train_loss}')
    
            if (epoch + 1) % 10 == 0:
                current_psnr, current_ssim = self.validation()
                print(f'Epoch: {epoch+1}, PSNR: {current_psnr}, SSIM: {current_ssim}')
                if current_psnr > self.best_psnr:
                    self.best_psnr = current_psnr
                    if not os.path.exists(self.ckpt_dir):
                        os.makedirs(self.ckpt_dir)
                    torch.save(self.network.state_dict(), os.path.join(self.ckpt_dir, 'pretrained_RRDBNet.pth'))

    def validation(self):
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        psnr_vals = []
        ssim_vals = []
        
        self.network.eval()
        with torch.no_grad():
            for lr_img, hr_img in self.valid_loader:
                lr_img, hr_img = lr_img.to(self.device), hr_img.to(self.device)
                output = self.network(lr_img)
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