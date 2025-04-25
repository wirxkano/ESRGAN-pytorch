import os
import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from constants import *
from Dataset import ImageDataset
from Train.SRTrainer import SRTrainer
from Train.SRGANTrainer import SRGANTrainer
from Model.RRDBNet import RRDBNet
from Model.VGGDiscriminator import VGGDiscriminator

def main():
    parser = argparse.ArgumentParser(description="ESRGAN: Train or Inference Mode")
    parser.add_argument('--mode', type=str, choices=['train_rrdb', 'train_esrgan', 'test', 'inference'], required=True,
                      help="Mode to run: 'train_rrdb', 'train_esrgan', 'test' or 'inference'")

    parser.add_argument('--rrdbnet_ckpt_dir', type=str, default=None, help="Directory storing RRDBNet checkpoint")
    parser.add_argument('--esrgan_ckpt_dir', type=str, default=None, help="Directory storing ESRGAN checkpoint")
    parser.add_argument('--rrdbnet_ckpt_path', type=str, default=None, help="Path to the checkpoint file (.pth)")
    parser.add_argument('--esrgan_ckpt_path', type=str, default=None, help="Path to the checkpoint file (.pth)")
    parser.add_argument('--train_dir', type=str, default=None, help="Directory storing train dataset")
    parser.add_argument('--valid_dir', type=str, default=None, help="Directory storing validation dataset")
    parser.add_argument('--test_dir', type=str, default=None, help="Directory storing test dataset")
    parser.add_argument('--img_path', type=str, default=None, help="Path to the image to process")
    
    args = parser.parse_args()
    
    if args.mode == 'train_rrdb':
        if not args.rrdbnet_ckpt_dir:
            parser.error("--rrdbnet_ckpt_dir is required for train mode (can be empty folder)")
        
        train_dataset = ImageDataset(args.train_dir)
        valid_dataset = ImageDataset(args.valid_dir, train=False)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=2, shuffle=False)
        
        g_model = SRTrainer(RRDBNet(3, 3, 64, 32, 23), train_loader, valid_loader, args.rrdbnet_ckpt_dir)
        g_model.train()
        
    elif args.mode == 'train_esrgan':
        if not args.rrdbnet_ckpt_path or not args.esrgan_ckpt_dir:
            parser.error("--esrgan_ckpt_dir (can be empty folder) and --rrdbnet_ckpt_path is required for train model")
        
        train_dataset = ImageDataset(args.train_dir)
        valid_dataset = ImageDataset(args.valid_dir, train=False)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=2, shuffle=False)
        
        g_model = SRTrainer(RRDBNet(3, 3, 64, 32, 23), None, None)
        g_model.network.load_state_dict(torch.load(args.rrdbnet_ckpt_path, weights_only=True))
        esrgan_model = SRGANTrainer(g_model.network, VGGDiscriminator(in_channels=3, num_features=64), train_loader, valid_loader, args.esrgan_ckpt_dir)
        
        if not args.esrgan_ckpt_path:
            start_epoch = 0
        else:
            start_epoch = esrgan_model.load_checkpoint(args.esrgan_ckpt_path)
            
        esrgan_model.train(start_epoch=start_epoch)
        
    elif args.mode == 'test':
        if not args.esrgan_ckpt_path or not args.rrdbnet_ckpt_path or not args.test_dir:
            parser.error("--esrgan_ckpt_path, --rrdbnet_ckpt_path and --test_dir are required for test mode")
        
        test_dataset = ImageDataset(args.test_dir, train=False)
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=2, shuffle=False)
        
        g_model = SRTrainer(RRDBNet(3, 3, 64, 32, 23), None, None)
        g_model.network.load_state_dict(torch.load(args.rrdbnet_ckpt_path, weights_only=True))
        esrgan_model = SRGANTrainer(g_model.network, VGGDiscriminator(in_channels=3, num_features=64), None, None)
        esrgan_model.load_checkpoint(args.esrgan_ckpt_path)
        esrgan_model.validation(test_loader)
    
    elif args.mode == 'inference':
        if not args.esrgan_ckpt_path or not args.rrdbnet_ckpt_path or not args.img_path:
            parser.error("--esrgan_ckpt_path, --rrdbnet_ckpt_path and --img_path are required for inference mode")
            
        if not os.path.exists(args.img_path):
            parser.error(f"Image path {args.img_path} does not exist.")
        
        g_model = SRTrainer(RRDBNet(3, 3, 64, 32, 23), None, None)
        g_model.network.load_state_dict(torch.load(args.rrdbnet_ckpt_path, weights_only=True))
        esrgan_model = SRGANTrainer(g_model.network, VGGDiscriminator(in_channels=3, num_features=64), None, None)
        esrgan_model.load_checkpoint(args.esrgan_ckpt_path)
        
        output = esrgan_model.inference(args.img_path)
        if isinstance(output, torch.Tensor):
            output = output.cpu().detach()
            if output.dim() == 4:
                output = output.squeeze(0)
            
            img = transforms.ToPILImage()(output)
            img.save('result.png')
            print("Inference completed. Saved to result.png")
        else:
            print("Error: Unexpected output type from inference.")
  
  
if __name__ == '__main__':
    main()
  