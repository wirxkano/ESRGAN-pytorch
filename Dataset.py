import os
from PIL import Image
from constants import *
from torchvision import transforms
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, hr_dir, scale=4, train=True):
        self.hr_dir = hr_dir
        self.image_files = sorted(os.listdir(self.hr_dir))
        self.image_length = len(self.image_files)
        self.scale = scale
        self.train = train
        
        self.train_preprocess = transforms.Compose([
            transforms.RandomCrop(PATCH_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((0, 90))
        ])

    def __getitem__(self, index):
        hr_path = os.path.join(self.hr_dir, self.image_files[index])
        hr_img = Image.open(hr_path).convert('RGB')

        hr_img = hr_img.crop((0, 0, hr_img.width - hr_img.width % self.scale, hr_img.height - hr_img.height % self.scale))
        
        if self.train:
            hr_img = self.train_preprocess(hr_img)

        lr_img = hr_img.resize((hr_img.width // self.scale, hr_img.height // self.scale), Image.BICUBIC)
        hr_img = transforms.ToTensor()(hr_img)
        lr_img = transforms.ToTensor()(lr_img)

        return lr_img, hr_img

    def __len__(self):
        return self.image_length
      