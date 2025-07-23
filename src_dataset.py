import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CelebAColorizationDataset(Dataset):
    def __init__(self, image_folder, image_size=128):
        self.image_folder = image_folder
        self.image_filenames = sorted([
            fname for fname in os.listdir(image_folder) if fname.endswith(".jpg")
        ])
        
        self.transform_input = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=1),  # Input: Grayscale
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])

        self.transform_target = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")
        gray = self.transform_input(image)
        color = self.transform_target(image)
        return gray, color