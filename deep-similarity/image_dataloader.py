import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import glob
class ImageDataSet(Dataset):   
    def __init__(self, data_path: str, pair: str = "stereo",transform=None):
        """
        Args:
            data_path: Images directory path
            pair: "stereo" (left/right) or other (sequential)
            transform: Optional preprocessing
        """
        self.pair = pair
        self.left_files = sorted(glob.glob(os.path.join(data_path, 'left_stereo', '*.jpeg')))  
        self.right_files = sorted(glob.glob(os.path.join(data_path, 'right_stereo', '*.jpeg'))) if pair == "stereo" else None
        
        if not self.left_files:
            raise ValueError("No images found in the left_stereo folder.")
        if pair == "stereo" and (not self.right_files or len(self.left_files) != len(self.right_files)):
            raise ValueError("Mismatch in left and right stereo image pairs.")

        if transform==None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform=transform

    def __len__(self):
        """Returns dataset size"""
        return len(self.left_files) if self.pair == 'stereo' else len(self.left_files) - 1

    def __getitem__(self, idx:int):    
        """
        Returns: torch.Tensor [2, C, H, W] - stacked image pair
        """
        left_img = self.load_image(self.left_files[idx])
        
        if self.pair == 'stereo':
            right_img = self.load_image(self.right_files[idx])
            return torch.stack([left_img, right_img], dim=0)
        else:
            next_img = self.load_image(self.left_files[idx + 1])
            return torch.stack([left_img, next_img], dim=0)

    def load_image(self, path:str):
        img = Image.open(path).convert("RGB")
        return self.transform(img)

def main():
    data_path = "/home/harish/noteworthy_project/data"
    pair = "stereo"  # Use 'stereo' or 'adjacent'
    
    dataset = ImageDataSet(data_path=data_path, pair=pair)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    for i, batch in enumerate(dataloader):
        print(f"Batch {i + 1}: Image tensor shape: {batch.shape}")
        break

if __name__ == "__main__":
    main()
