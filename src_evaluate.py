import torch
from torchvision.utils import save_image
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from dataset import CelebAColorizationDataset
from models import UNetGenerator

def evaluate_gan(image_folder, batch_size=128, image_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = CelebAColorizationDataset(image_folder, image_size)
    total_len = len(dataset)
    train_len = total_len // 2
    test_len = total_len - train_len
    _, test_dataset = torch.utils.data.random_split(dataset, [train_len, test_len])
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Load generator
    generator = UNetGenerator().to(device)
    generator.load_state_dict(torch.load("generator_final.pth", map_location=device))
    generator.eval()

    # Output folder
    os.makedirs("results", exist_ok=True)

    # Colorfulness metric
    def compute_colorfulness(image_np):
        R, G, B = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]
        rg = np.abs(R - G)
        yb = np.abs(0.5 * (R + G) - B)
        std_rg, std_yb = np.std(rg), np.std(yb)
        mean_rg, mean_yb = np.mean(rg), np.mean(yb)
        return np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)

    # Track metrics
    ssim_scores, psnr_scores, colorfulness_scores = [], [], []

    with torch.no_grad():
        for idx, (gray_img, real_img) in enumerate(testloader):
            gray_img = gray_img.to(device)
            real_img = real_img.to(device)
            fake_img = generator(gray_img)

            for b in range(gray_img.size(0)):
                gray = gray_img[b]
                fake = fake_img[b]
                real = real_img[b]
                combined = torch.cat([gray.repeat(3, 1, 1), fake, real], dim=2)
                save_image(combined, f"results/sample_{idx:03d}_{b}.png", normalize=True)

                fake_np = fake.permute(1, 2, 0).cpu().numpy()
                real_np = real.permute(1, 2, 0).cpu().numpy()
                fake_np = np.clip(fake_np, 0, 1)
                real_np = np.clip(real_np, 0, 1)

                ssim_val = ssim(real_np, fake_np, channel_axis=2, data_range=1.0)
                psnr_val = psnr(real_np, fake_np, data_range=1.0)
                colorfulness_val = compute_colorfulness(fake_np)

                ssim_scores.append(ssim_val)
                psnr_scores.append(psnr_val)
                colorfulness_scores.append(colorfulness_val)

    print(f"Avg SSIM: {np.mean(ssim_scores):.4f}")
    print(f"Avg PSNR: {np.mean(psnr_scores):.2f} dB")
    print(f"Avg Colorfulness: {np.mean(colorfulness_scores):.2f}")

if __name__ == "__main__":
    image_folder = "/kaggle/input/face-vae/img_align_celeba/img_align_celeba"
    evaluate_gan(image_folder)