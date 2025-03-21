import argparse
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from PIL import Image
import torchvision
from torchvision.utils import save_image
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, CLIPVisionModel
import numpy as np
from typing import List, Tuple, Optional
import functools

# VAE برای فضای نهان
class VAEEncoder(nn.Module):
    def __init__(self, in_ch=3, hidden_ch=256, z_dim=128, resolution=512):
        super().__init__()
        self.resolution = resolution
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch * 2, 4, stride=2, padding=1)
        reduced_size = resolution // 4
        self.fc_mu = nn.Linear(hidden_ch * 2 * reduced_size * reduced_size, z_dim)
        self.fc_logvar = nn.Linear(hidden_ch * 2 * reduced_size * reduced_size, z_dim)
        self.bn1 = nn.BatchNorm2d(hidden_ch)
        self.bn2 = nn.BatchNorm2d(hidden_ch * 2)
    
    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class VAEDecoder(nn.Module):
    def __init__(self, z_dim=128, hidden_ch=256, out_ch=3, resolution=512):
        super().__init__()
        self.resolution = resolution
        reduced_size = resolution // 4
        self.fc = nn.Linear(z_dim, hidden_ch * 2 * reduced_size * reduced_size)
        self.conv1 = nn.ConvTranspose2d(hidden_ch * 2, hidden_ch, 4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(hidden_ch, out_ch, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_ch)
    
    def forward(self, z):
        h = self.fc(z).view(-1, 256 * 2, self.resolution // 4, self.resolution // 4)
        h = F.relu(self.bn1(self.conv1(h)))
        h = torch.tanh(self.conv2(h))
        return h

# U-Net پیشرفته با Self-Attention
class UNetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, upsample: bool = False, downsample: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.self_attn = nn.MultiheadAttention(out_ch, 8)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear') if upsample else nn.Identity()
        self.down = nn.AvgPool2d(2) if downsample else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        x = self.up(x)
        x = self.act(self.bn1(self.conv1(x)))
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(2, 0, 1)
        x_flat, _ = self.self_attn(x_flat, x_flat, x_flat)
        x = x_flat.permute(1, 2, 0).view(B, C, H, W)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.down(x)
        return x

# Generator با Progressive Growing و StyleGAN
class Generator(nn.Module):
    def __init__(self, G_ch=64, dim_z=128, resolution=512, stages=[64, 128, 256, 512]):
        super().__init__()
        self.ch = G_ch
        self.dim_z = dim_z
        self.resolution = resolution
        self.stages = stages
        self.bottom_width = 4
        
        # StyleGAN
        self.style_fc = nn.Linear(dim_z + 1024, 512)
        self.style_blocks = nn.ModuleList([nn.Linear(512, G_ch * 2) for _ in range(len(stages))])
        
        # Progressive Growing
        self.blocks = nn.ModuleList()
        current_ch = G_ch * 16
        for i, res in enumerate(stages):
            block = UNetBlock(current_ch, G_ch * (8 // (i + 1)), upsample=True)
            self.blocks.append(block)
            current_ch = G_ch * (8 // (i + 1))
        
        self.to_rgb = nn.ModuleList([nn.Conv2d(current_ch, 3, 1) for _ in stages])
        self.current_stage = 0
    
    def forward(self, z, condition=None, stage=None):
        if stage is None:
            stage = self.current_stage
        style = self.style_fc(z)
        h = style.view(-1, self.ch * 16, self.bottom_width, self.bottom_width)
        
        for i, block in enumerate(self.blocks[:stage + 1]):
            h = block(h)
            style_vec = self.style_blocks[i](style)
            h = h * style_vec[:, :h.shape[1], None, None] + style_vec[:, h.shape[1]:, None, None]
        
        return torch.tanh(self.to_rgb[stage](h))
    
    def grow(self):
        self.current_stage = min(self.current_stage + 1, len(self.stages) - 1)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, D_ch=64, resolution=512, stages=[64, 128, 256, 512]):
        super().__init__()
        self.ch = D_ch
        self.resolution = resolution
        self.stages = stages
        
        self.blocks = nn.ModuleList()
        current_ch = 3
        for i, res in enumerate(stages[::-1]):
            block = UNetBlock(current_ch, D_ch * (2 ** i), downsample=True)
            self.blocks.append(block)
            current_ch = D_ch * (2 ** i)
        
        self.linear = nn.Linear(current_ch * 4 * 4, 1)
        self.current_stage = len(stages) - 1
    
    def forward(self, x):
        h = x
        for block in self.blocks[:self.current_stage + 1]:
            h = block(h)
        h = h.view(h.size(0), -1)
        return self.linear(h)
    
    def shrink(self):
        self.current_stage = max(self.current_stage - 1, 0)

# Diffusion
class DiffusionProcess:
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alphas = torch.sqrt(1 - self.alphas_cumprod[t])[:, None, None, None]
        return sqrt_alphas_cumprod * x + sqrt_one_minus_alphas * noise, noise
    
    def sample_step(self, model, x, t, condition):
        t_tensor = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
        pred_noise = model(x, t_tensor, condition)
        alpha = self.alphas[t][:, None, None, None]
        beta = self.betas[t][:, None, None, None]
        alpha_cumprod = self.alphas_cumprod[t][:, None, None, None]
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = (x - (1 - alpha) / torch.sqrt(1 - alpha_cumprod) * pred_noise) / torch.sqrt(alpha) + torch.sqrt(beta) * noise
        return x

# مدل اصلی
class AdvancedTextToImageModel(pl.LightningModule):
    def __init__(self, G_ch=64, D_ch=64, dim_z=128, resolution=512, timesteps=1000, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.vae_encoder = VAEEncoder(resolution=resolution)
        self.vae_decoder = VAEDecoder(resolution=resolution)
        self.generator = Generator(G_ch=G_ch, dim_z=dim_z, resolution=resolution)
        self.discriminator = Discriminator(D_ch=D_ch, resolution=resolution)
        self.diffusion = DiffusionProcess(timesteps=timesteps)
        
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
        self.text_encoder = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        
        self.automatic_optimization = False
    
    def forward(self, x_or_z, text_or_t, condition=None):
        if isinstance(text_or_t, list):
            text = text_or_t
            z = x_or_z
            text_tokens = self.tokenizer(text, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
            text_embeds = self.text_encoder(**text_tokens.to(self.device)).last_hidden_state[:, 0, :]
            z_combined = torch.cat([z, text_embeds], dim=1)
            return self.generator(z_combined, condition=self.generator.current_stage)
        else:
            x, t = x_or_z, text_or_t
            text_tokens = self.tokenizer(condition, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
            text_embeds = self.text_encoder(**text_tokens.to(self.device)).last_hidden_state[:, 0, :]
            z = self.vae_encoder.reparameterize(*self.vae_encoder(x))
            z_combined = torch.cat([z, text_embeds], dim=1)
            return self.generator(z_combined)
    
    def gradient_penalty(self, real_imgs, fake_imgs):
        alpha = torch.rand(real_imgs.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs.detach()).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.ones_like(d_interpolates)
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake, create_graph=True)[0]
        gp = ((gradients.norm(2, dim=[1, 2, 3]) - 1) ** 2).mean()
        return gp
    
    def training_step(self, batch, batch_idx):
        real_imgs, texts = batch
        batch_size = real_imgs.size(0)
        
        opt_g, opt_d, opt_vae = self.optimizers()
        
        mu, logvar = self.vae_encoder(real_imgs)
        z = self.vae_encoder.reparameterize(mu, logvar)
        recon_imgs = self.vae_decoder(z)
        vae_recon_loss = F.mse_loss(recon_imgs, real_imgs)
        vae_kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = vae_recon_loss + 0.1 * vae_kl_loss
        
        t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device)
        noisy_imgs, noise = self.diffusion.add_noise(real_imgs, t)
        
        opt_d.zero_grad()
        fake_z = torch.randn(batch_size, 128, device=self.device)
        fake_imgs = self(fake_z, texts)
        real_validity = self.discriminator(real_imgs)
        fake_validity = self.discriminator(fake_imgs.detach())
        gp = self.gradient_penalty(real_imgs, fake_imgs)
        d_loss = torch.mean(F.relu(1. - real_validity)) + torch.mean(F.relu(1. + fake_validity)) + 10 * gp
        self.manual_backward(d_loss)
        opt_d.step()
        
        opt_g.zero_grad()
        pred_noise = self(noisy_imgs, t, texts)
        g_diff_loss = F.mse_loss(pred_noise, noise)
        fake_validity = self.discriminator(fake_imgs)
        g_gan_loss = -torch.mean(fake_validity)
        g_loss = g_diff_loss + 0.1 * g_gan_loss
        self.manual_backward(g_loss)
        opt_g.step()
        
        opt_vae.zero_grad()
        self.manual_backward(vae_loss)
        opt_vae.step()
        
        self.log_dict({"d_loss": d_loss, "g_loss": g_loss, "vae_loss": vae_loss}, prog_bar=True)
        if batch_idx % 1000 == 0 and self.generator.current_stage < len(self.generator.stages) - 1:
            self.generator.grow()
            self.discriminator.shrink()
    
    def configure_optimizers(self):
        opt_g = optim.AdamW(self.generator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999), weight_decay=0.01)
        opt_d = optim.AdamW(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999), weight_decay=0.01)
        opt_vae = optim.AdamW(list(self.vae_encoder.parameters()) + list(self.vae_decoder.parameters()), lr=self.hparams.lr)
        return [opt_g, opt_d, opt_vae], []
    
    def generate_image(self, text: str, num_samples: int = 1, steps: int = 50, resolution: int = 512):
        self.generator.resolution = resolution
        z = torch.randn(num_samples, 128, device=self.device)
        x = torch.randn(num_samples, 3, resolution, resolution, device=self.device)
        
        with torch.no_grad():
            for t in reversed(range(steps)):
                x = self.diffusion.sample_step(self, x, t, [text] * num_samples)
            x = self.vae_decoder(self.vae_encoder.reparameterize(*self.vae_encoder(x)))
        return x
    
    def edit_image(self, image: torch.Tensor, text: str, steps: int = 50):
        image = image.to(self.device)
        mu, logvar = self.vae_encoder(image)
        z = self.vae_encoder.reparameterize(mu, logvar)
        x = self(z, [text])
        with torch.no_grad():
            for t in reversed(range(steps)):
                x = self.diffusion.sample_step(self, x, t, [text])
        return self.vae_decoder(self.vae_encoder.reparameterize(*self.vae_encoder(x)))

# دیتاست LAION
class LAIONDataset(Dataset):
    def __init__(self, data_dir: str, resolution: int = 512):
        self.data_dir = data_dir
        self.resolution = resolution
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((resolution, resolution)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ])
        # فرض می‌کنیم LAION به صورت جفت تصویر-متن در data_dir ذخیره شده
        self.data = self.load_laion_data()
    
    def load_laion_data(self) -> List[Tuple[Image.Image, str]]:
        # اینجا باید کد واقعی برای لود LAION پیاده‌سازی بشه
        # فعلاً dummy برای تست
        return [(Image.new('RGB', (self.resolution, self.resolution)), f"image_{i}") for i in range(100)]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img, text = self.data[idx]
        return self.transform(img), text

class TextImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 2, resolution: int = 512):  # batch_size کاهش داده شده
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.resolution = resolution
    
    def setup(self, stage: Optional[str] = None):
        self.train_dataset = LAIONDataset(self.data_dir, self.resolution)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

# کال‌بک
class ImageLogger(pl.Callback):
    def __init__(self, log_every_n_steps: int = 100):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_steps == 0:
            with torch.no_grad():
                img = pl_module.generate_image("A cat in the forest", num_samples=1)
                trainer.logger.experiment.add_image("generated", img[0], trainer.global_step)

# تابع اصلی
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="یه گربه تو جنگل", help="Text prompt")
    parser.add_argument("--resolution", type=int, default=512, help="Output resolution")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--data_dir", type=str, default="path/to/laion", help="Path to LAION dataset")
    parser.add_argument("--edit_image", type=str, default=None, help="Path to image for editing")
    args = parser.parse_args()

    model = AdvancedTextToImageModel(resolution=args.resolution)
    dm = TextImageDataModule(data_dir=args.data_dir, batch_size=args.batch_size, resolution=args.resolution)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=1 if torch.cuda.is_available() else 0,
        precision=16 if torch.cuda.is_available() else 32,
        callbacks=[ImageLogger()],
        accelerator='ddp' if torch.cuda.device_count() > 1 else None,
        gradient_clip_val=1.0
    )
    
    trainer.fit(model, dm)
    
    if args.edit_image:
        img = Image.open(args.edit_image).convert("RGB")
        img_tensor = torchvision.transforms.ToTensor()(img).unsqueeze(0).to(model.device)
        edited_img = model.edit_image(img_tensor, args.text)
        save_image(edited_img, "edited_output.png", normalize=True)
    else:
        img = model.generate_image(args.text, num_samples=1, resolution=args.resolution)
        save_image(img, "output.png", normalize=True)

if __name__ == "__main__":
    main()

# لایسنس و منابع
"""
MIT License

Copyright (c) 2025 Hossein Davoodabadi Farahani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Inspiration Sources:
1. BigGAN: "Large Scale GAN Training for High Fidelity Natural Image Synthesis" (Brock et al., 2018)
   - GitHub: https://github.com/ajbrock/BigGAN-PyTorch
2. Stable Diffusion: "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al., 2022)
   - GitHub: https://github.com/CompVis/stable-diffusion
3. StyleGAN: "A Style-Based Generator Architecture for Generative Adversarial Networks" (Karras et al., 2019)
   - GitHub: https://github.com/NVlabs/stylegan
4. Progressive GANs: "Progressive Growing of GANs for Improved Quality, Stability, and Variation" (Karras et al., 2017)
   - GitHub: https://github.com/tkarras/progressive_growing_of_gans
5. mBART: "Multilingual Denoising Pre-training for Neural Machine Translation" (Liu et al., 2020)
   - GitHub: https://github.com/facebookresearch/fairseq
6. LAION: Large-scale Artificial Intelligence Open Network
   - Website: https://laion.ai/
7. PyTorch Lightning: Framework for scalable deep learning
   - GitHub: https://github.com/PyTorchLightning/pytorch-lightning
"""
