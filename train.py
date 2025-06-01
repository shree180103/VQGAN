import torch
from torch import nn
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader,Dataset
import torchvision
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from PIL import Image
import numpy as np
import os
# from perceptual_loss import LPIPS
from models import VQGAN,Discriminator
import utils
from utils import Data_pipeline,weights_init,save_image
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')


class Train_VQGAN:
    def __init__(self,args):
        
        self.device=args.device
        self.VQGAN=VQGAN(args).to(self.device)
        self.disc=Discriminator(args).to(self.device)
        # self.perceptual_loss=LPIPS().eval().to(self.device)

        self.optimizer_VQ,self.optimizer_disc=self.Configure_optimizers(args)
        
        self.prepare_training()

        self.train(args)

    def Configure_optimizers(self,args):

        lr=args.learning_rate

        optimizer_VQ=torch.optim.Adam(list(self.VQGAN.encoder.parameters())+
                                      list(self.VQGAN.decoder.parameters())+
                                      list(self.VQGAN.codebook.parameters())+
                                      list(self.VQGAN.pre_quant_conv.parameters())+
                                      list(self.VQGAN.post_quant_conv.parameters()),
                                      lr=lr,eps=1e-08, betas=(args.beta1,args.beta2)
                                      
                                      )
        
        optimizer_disc=torch.optim.Adam(self.disc.parameters(),
                                      lr=lr,eps=1e-08, betas=(args.beta1,args.beta2)
                                      
                                      )
        
        return optimizer_VQ,optimizer_disc
    
    
    @staticmethod
    def prepare_training():
        os.makedirs("results",exist_ok=True)
        os.makedirs("checkpoint",exist_ok=True)

    
    def train(self,args):
        train_dataloader=Data_pipeline(args)
        steps_per_epoch=len(train_dataloader)

        for epoch in range(args.epoch):
            with tqdm(range(len(train_dataloader))) as pbar:
                for i,imgs in zip(pbar,train_dataloader):

                    imgs=imgs[0].to(device=args.device)

                    recon_img,vq_loss,indices=self.VQGAN(imgs)

                    disc_real=self.disc(imgs)
                    disc_fake=self.disc(recon_img.detach())
                    disc_factor=self.VQGAN.adapt_weights(args.disc_factor,epoch*steps_per_epoch + 1,threshold=args.disc_start)

                    # perceptual_loss=self.perceptual_loss(imgs,recon_img)
                    recon_loss=(imgs - recon_img)**2
                    # total_recon_loss=(args.perceptual_weight*perceptual_loss + args.recon_weight*recon_loss).mean()
                    total_recon_loss=(recon_loss).mean()

                    disc_loss=disc_factor*(torch.mean(F.relu(1-disc_real)) + torch.mean(F.relu(1+ disc_fake)))/2
                    self.optimizer_disc.zero_grad()
                    disc_loss.backward()
                    self.optimizer_disc.step()

                    self.optimizer_VQ.zero_grad()

                    fake_logits=self.disc(recon_img.detach())
                    gan_loss=-torch.mean(fake_logits)

                    lambda_=self.VQGAN.lambda_(total_recon_loss,gan_loss)
                    total_VQ_loss=  total_recon_loss + vq_loss + disc_factor*lambda_*gan_loss
                    total_VQ_loss.backward()
                    self.optimizer_VQ.step()

                    if i%10==0:
                        with torch.no_grad():
                            real_fake_images=torch.cat([imgs[:4],recon_img[:4]],dim=0)
                            utils.save_image(real_fake_images,os.path.join("results",f"epoch_{epoch}_step_{i}.png"),nrow=4)

                            pbar.set_description(f"Epoch {epoch} Step {i} VQ Loss: {vq_loss.item():.4f} Disc Loss: {disc_loss.item():.4f} Recon Loss: {total_recon_loss.item():.4f} GAN Loss: {gan_loss.item():.4f}")
                            pbar.set_postfix({"VQ Loss": vq_loss.item(),
                                              "Disc Loss": disc_loss.item(),
                                              "Recon Loss": total_recon_loss.item(),
                                              "GAN Loss": gan_loss.item()})
                            
                            pbar.update(0)

                        torch.save(self.VQGAN.state_dict(),os.path.join("checkpoint",f"VQGAN_epoch_{epoch}_step_{i}.pth"))
                        torch.save(self.disc.state_dict(),os.path.join("checkpoint",f"disc_epoch_{epoch}_step_{i}.pth"))
                                           


                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=" VQGAN")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--epoch", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=2.25e-05, help="Learning rate for optimizers")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.9, help="Beta2 for Adam optimizer")
    # parser.add_argument("--perceptual_weight", type=float, default=1, help="Weight for perceptual loss")
    # parser.add_argument("--recon_weight", type=float, default=1, help="Weight for reconstruction loss")
    parser.add_argument("--disc_factor", type=float, default=1, help="Discriminator factor to start with")
    parser.add_argument("--disc_start", type=int, default=10000, help="Epoch after which discriminator starts training")
    parser.add_argument("--image_size", type=int, default=256, help="Size of the input images")
    # parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--latent_dim", type=int, default=256, help="Dimension of the latent space")
    parser.add_argument("--k", type=int, default=512, help="Number of embeddings in the codebook")
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument("--in_channels", type=int, default=3, help="Number of channels in the input images")
    parser.add_argument("--out_channels", type=int, default=3, help="Input shape for the discriminator")
    parser.add_argument("--path", type=str, default="C:/Users/shree/OneDrive/Documents/VQGAN/scenes/", help="Path to the dataset")

    args = parser.parse_args()
    # args.dataset_path=r"C:/Users/shree/OneDrive/Documents/VQGAN/scenes/"
    trainer = Train_VQGAN(args)

        