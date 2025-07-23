import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision
import matplotlib.pyplot as plt
from dataset import CelebAColorizationDataset
from models import UNetGenerator, PatchDiscriminator
from losses import VGGPerceptualLoss

def train_gan(image_folder, epochs=20, batch_size=128, image_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset = CelebAColorizationDataset(image_folder, image_size)
    total_len = len(dataset)
    train_len = total_len // 2
    test_len = total_len - train_len
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_len, test_len])
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Models and optimizers
    generator = UNetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # Losses
    adversarial_loss = nn.BCEWithLogitsLoss()
    pixelwise_loss = nn.L1Loss()
    percep_loss = VGGPerceptualLoss(device)

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        loop = tqdm(dataloader, leave=True)
        
        for i, (gray_imgs, real_imgs) in enumerate(loop):
            gray_imgs = gray_imgs.to(device)
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # Train Discriminator
            optimizer_D.zero_grad()
            pred_real = discriminator(gray_imgs, real_imgs)
            valid = torch.ones_like(pred_real, device=device)
            d_real_loss = adversarial_loss(pred_real, valid)
            fake_imgs = generator(gray_imgs)
            pred_fake = discriminator(gray_imgs, fake_imgs.detach())
            fake = torch.zeros_like(pred_fake, device=device)
            d_fake_loss = adversarial_loss(pred_fake, fake)
            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            pred_fake = discriminator(gray_imgs, fake_imgs)
            valid = torch.ones_like(pred_fake, device=device)
            g_adv_loss = adversarial_loss(pred_fake, valid)
            g_pixel_loss = pixelwise_loss(fake_imgs, real_imgs)
            g_percep_loss = percep_loss(fake_imgs, real_imgs)
            g_loss = g_adv_loss + 100 * g_pixel_loss + 0.1 * g_percep_loss
            g_loss.backward()
            optimizer_G.step()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(D_Loss=d_loss.item(), G_Loss=g_loss.item())

        # Visualize samples
        generator.eval()
        with torch.no_grad():
            for test_gray, test_real in testloader:
                test_gray = test_gray.to(device)
                test_real = test_real.to(device)
                sample_gray = test_gray[:8]
                sample_real = test_real[:8]
                sample_fake = generator(sample_gray)
                sample_gray_3c = sample_gray.expand(-1, 3, -1, -1)
                samples = torch.cat([sample_gray_3c, sample_fake, sample_real], dim=0)
                grid = torchvision.utils.make_grid(samples, nrow=8, normalize=True)
                plt.figure(figsize=(15, 5))
                plt.title(f"Epoch {epoch+1} - Grayscale | Generated | Real")
                plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
                plt.axis("off")
                plt.show()
                break

    # Save models
    torch.save(generator.state_dict(), "generator_final.pth")
    torch.save(discriminator.state_dict(), "discriminator_final.pth")

if __name__ == "__main__":
    image_folder = "/kaggle/input/face-vae/img_align_celeba/img_align_celeba"
    train_gan(image_folder)