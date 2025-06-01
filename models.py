import torch 
import torch.nn as nn
import torch.nn.functional as F
from helper import GroupNorm, swish, ResidualBlock, UpsampleBlock, DownsampleBlock, NonLocalBlock

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args=args
        channels = [128, 128, 128, 256, 256, 512]
        attn_resolution=[16]
        m=2
        resolution=256

        layers = [nn.Conv2d(args.in_channels, channels[0], kernel_size=3, stride=1, padding=1),]
        for i in range(len(channels)-1):
            input_channels = channels[i]
            output_channels = channels[i+1]
            for j in range(m):
                layers.append(ResidualBlock(input_channels, output_channels))
                input_channels = output_channels

                if resolution in attn_resolution:
                    layers.append(NonLocalBlock(channels[i+1]))

            if i < len(channels)-2:
                layers.append(UpsampleBlock(channels[i+1]))
                resolution = resolution // 2

        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(NonLocalBlock(channels[-1]))
        layers.append(ResidualBlock(channels[-1], channels[-1]))
        layers.append(GroupNorm(channels[-1]))
        layers.append(swish())  
        layers.append(nn.Conv2d(channels[-1], args.latent_dim, kernel_size=3, stride=1, padding=1))
        self.model=nn.Sequential(*layers)


        

    def forward(self, x):
    
        return self.model(x)
    


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args=args
        channels = [512, 256, 256, 128, 128, 128]
        attn_resolution=[16]
        resolution=16
        m=3
         
        in_channels = channels[0]

        layers = [nn.Conv2d(args.latent_dim, channels[0], kernel_size=3, stride=1, padding=1),
                  ResidualBlock(channels[0], channels[0]),
                  NonLocalBlock(channels[0]),   
                  ResidualBlock(channels[0], channels[0]),]
        
        for i in range(len(channels)-1):
            out_channels = channels[i]
            layers.append(ResidualBlock(in_channels,out_channels))
            in_channels = out_channels
            if resolution in attn_resolution:
                layers.append(NonLocalBlock(in_channels))
            if i !=0:
                layers.append(UpsampleBlock(in_channels))
                resolution = resolution * 2

        layers.append(GroupNorm(in_channels))
        layers.append(swish())  
        layers.append(nn.Conv2d(in_channels, args.out_channels, kernel_size=3, stride=1, padding=1))
        self.model=nn.Sequential(*layers)


    def forward(self, x):
    
        return self.model(x)
    

class Codebook(nn.Module):
    def __init__(self, args):
        super(Codebook, self).__init__()
        self.args=args
        self.k= args.k
        self.embedding_dim = args.latent_dim
        self.beta= args.beta
        self.embeded_space= nn.Embedding(self.k, self.embedding_dim)
        self.embeded_space.weight.data.uniform_(-1/self.k, 1/self.k)

    def forward(self, z):
        z =z.permute(0, 2, 3, 1).contiguous()  # Change to (batch_size, height, width, channels)
        z_flattened = z.view(-1, self.embedding_dim)
        # l2_norm expanded into a2+b2-2ab as z is (4096, 64) and embeded_space.weight is (512, 64) thus on exapnsion (4096, 1) and (1, 512) respectively thus can be added by bradcasting 
        l2_norm= torch.sum(z_flattened ** 2, dim=1, keepdim=True)+torch.sum(self.embeded_space.weight ** 2, dim=1) - 2 * torch.matmul(z_flattened, self.embeded_space.weight.t())
        min_indices = torch.argmin(l2_norm, dim=1)
        z_q= self.embeded_space(min_indices).view(z.shape)

        loss = torch.mean((z.detach()   - z_q)**2) + self.beta * torch.mean((z - z_q.detach())**2)

        # preserve gradients
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, min_indices
    



class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.args=args
        self.encoder = Encoder(args).to(device=args.device)
        self.decoder = Decoder(args).to(device=args.device)
        self.codebook = Codebook(args).to(device=args.device)
        self.pre_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, kernel_size=1, stride=1, padding=0).to(device=args.device)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, kernel_size=1, stride=1, padding=0).to(device=args.device)

    def encode(self, x):
        z_e=self.encoder(x)
        z_e=self.pre_quant_conv(z_e)
        z_q, loss, indices = self.codebook(z_e)
        return z_e, loss, indices   
    
    def decode(self, z_q):
        z_q = self.post_quant_conv(z_q)
        return self.decoder(z_q)
    

    def code_book(self, z_e):
        return self.codebook(z_e)
    

    def lambda_(self, recon_loss, gan_loss):
        last_layer=self.decoder[-1]
        last_layer_weight = last_layer.weight
        recon_loss_grads = torch.autograd.grad(recon_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        lambda_ = torch.mean(torch.abs(recon_loss_grads)) / (torch.mean(torch.abs(gan_loss_grads)) + 1e-6)
        lambda_ = torch.clamp(lambda_, 0, 1e4).detach()

        return 0.8*lambda_

    @staticmethod
    def adapt_weights(disc_factor,i,threshold,value=0.1):

        """Starting the discrimator later in training, so that our model has enough time to generate "good-enough" images to try to "fool the discrimator".

        To do that, we before eaching a certain global step, set the discriminator factor by `value` ( default 0.0 ) .
        This discriminator factor is then used to multiply the discriminator's loss.

        Args:
            disc_factor (float): This value is multiple to the discriminator's loss.
            i (int): The current global step
            threshold (int): The global step after which the `disc_factor` value is retured.
            value (float, optional): The value of discriminator factor before the threshold is reached. Defaults to 0.0.

        Returns:
            float: The discriminator factor.
        """

        if i < threshold:
            disc_factor=value
        
        return disc_factor
        

    def forward(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quant_conv(z_e)
        z_q, loss, indices = self.codebook(z_e)
        z_q = self.post_quant_conv(z_q)
        x_recon = self.decoder(z_q)
        return x_recon, loss, indices
    

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path, map_location=self.args.device))


    
class Discriminator(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.patch_gan=nn.Sequential(
            nn.Conv2d(in_channels=args.in_channels,out_channels=64,kernel_size=4,stride=2,padding=1,bias=False,padding_mode="reflect"),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4,stride=2,padding=1,bias=False,padding_mode="reflect"),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=4,stride=2,padding=1,bias=False,padding_mode="reflect"),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=4,stride=1,padding=1,bias=False,padding_mode="reflect"),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(in_channels=512,out_channels=1,kernel_size=4,stride=1,padding=1,bias=False,padding_mode="reflect"),

        )

    def forward(self,x):
        return self.patch_gan(x)

    


