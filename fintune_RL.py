import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
# from huggingface_hub import notebook_login
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
# For video display:
from PIL import Image
from torchvision import transforms as tfms
from tqdm.auto import tqdm
from transformers import logging
from unet import UNet
from dataset import *
import os
import argparse
import warnings
import torch
from pytorch_lightning.loggers import WandbLogger
from x_unet import XUnet
from nevaluate import nmetrics

warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logging.set_verbosity_error()

# Set device
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the autoencoder model which will be used to decode the latents into image space. 
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
vae = vae.to(torch_device)
def pil_to_latent(input_im):
    # Single image -> single latent in a batch (so size 1, 4, 64, 64)
    with torch.no_grad():
        # latent = vae.encode(tfms.ToTensor()(input_im).unsqueeze(0).to(torch_device)*2-1) # Note scaling
        latent = vae.encode(input_im*2-1) # Note scaling
    return 0.18215 * latent.latent_dist.sample()

def latents_to_pil(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

    

def initialize_weights(model, method='xavier'):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if method == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif method == 'kaiming':
                nn.init.kaiming_uniform_(m.weight)
            else:
                raise ValueError('Unknown initialization method: %s' % method)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)



class MyModel(pl.LightningModule):
    def __init__(self, log_path):
        super().__init__()
        # self.model = UNet(in_channels=4, out_channels=4).to(torch_device)
        # self.model = MyCNN(in_channels=4).to(torch_device)
        self.unet = XUnet(
                dim = 64,
                channels = 4,
                dim_mults = (2, 4, 8),
                nested_unet_depths = (4, 2, 1),     # nested unet depths, from unet-squared paper
                consolidate_upsample_fmaps = True,     # whether to consolidate outputs from all upsample blocks, used in unet-squared paper
                )
        self.unet = self.unet.to(torch_device)
        initialize_weights(self.unet)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(torch_device)
        self.log_path = log_path
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def forward(self, x):
        return self.unet(x)
    
    def train_dataloader(self):
        train_dataset = ImageDataset(os.path.join('/root/zhshen/UW_datasets/UIEB', "Train"), img_size=(512,512))
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        return train_loader

    def training_step(self, batch, batch_idx):
        uw_image, gt_image= batch['underwater_image'],batch['gt_image']
        another_batch = next(iter(self.train_dataloader()))
        another_uw_image, another_gt_image= another_batch['underwater_image'],another_batch['gt_image']
        bs = another_uw_image.shape[0]
        tmp_gt_image = gt_image.detach().cpu().numpy()
        tmp_gt_image = np.transpose(tmp_gt_image, (0, 2, 3, 1))
        another_gt_image= another_gt_image.detach().cpu().numpy()
        another_gt_image = np.transpose(another_gt_image, (0, 2, 3, 1))
        uciqe = torch.tensor(0).to(torch_device).float()
        uciqe_another = torch.tensor(0).to(torch_device).float()
        for i in range(bs):
            _, uciqe_t = nmetrics(tmp_gt_image[i])
            _, uciqe_a = nmetrics(another_gt_image[i])
            uciqe_another += torch.tensor(uciqe_a).to(torch_device)
            uciqe += torch.tensor(uciqe_t).to(torch_device)
        uciqe_another = uciqe_another/bs
        uciqe = uciqe/bs
        r = uciqe - uciqe_another
        
        encoded = pil_to_latent(uw_image)
        output_latents = self(encoded)
        denoised_images = vae.decode((1 / 0.18215) * output_latents).sample / 2 + 0.5 # range (0, 1)
        denoised_images = torch.clamp(denoised_images, 0, 1)
        loss = r*self.lpips(denoised_images, gt_image)

        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        uw_image, gt_image = batch['underwater_image'],batch['gt_image']
        image_name = batch['image_name']
        encoded = pil_to_latent(uw_image)
        output_latents = self(encoded)
        denoised_images = vae.decode((1 / 0.18215) * output_latents).sample / 2 + 0.5 # range (0, 1)
        denoised_images = torch.clamp(denoised_images, 0, 1)
        # loss = self.lpips(denoised_images, gt_image)
        # self.log("val_loss", loss)
        for i in range(len(image_name)):
            save_image(denoised_images[i], os.path.join(self.log_path, image_name[i]))



    # def predict_step(self, batch, batch_idx, dataloader_idx=None):
    #     uw_image, gt_image = batch['underwater_image'],batch['gt_image']
    #     image_name = batch['image_name']
    #     _trans, _atm, _GT = self(uw_image)
    #     _GT = torch.clamp(_GT, 0, 1)
    #     return _GT


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=3e-7, betas=(0.99, 0.999), eps=1e-08, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "loss"}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_set', type=str, default='uieb')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=25)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--save_path', type=str, default='./logs_RL')
    args = parser.parse_args()

    pl.seed_everything(123)
    # Initialize WandbLogger
    wandb_logger = WandbLogger(project='vae_enhance')
    dm = MyDataModule(args.data_set, args.batch_size, args.num_workers, args.image_size, args.data_path)
    n_epochs = args.n_epochs
    model = MyModel(args.save_path)
    # if os.path.exists(os.path.join(args.save_path,'model.pt')):
    #     model.load_state_dict(torch.load(os.path.join(args.save_path,'model.pt')))
    model.load_state_dict(torch.load(os.path.join('/root/zhshen/my_diffusion/logs_xunet/model.pt')))
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=n_epochs,
        check_val_every_n_epoch=1,
        logger=wandb_logger
    )
    trainer.fit(model, datamodule = dm)
    # predictions = trainer.predict(model, datamodule = dm)
    torch.save(model.state_dict(), os.path.join(args.save_path,'model.pt'))

    