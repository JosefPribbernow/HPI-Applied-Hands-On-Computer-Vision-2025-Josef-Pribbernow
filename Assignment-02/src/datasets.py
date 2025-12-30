import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class FusionDataset(Dataset):
    """
    Dataset for RGB-LiDAR fusion classification tasks.
    
    Args:
        root_dir (str): Root directory containing class folders
        start_idx (int): Starting index for data samples
        stop_idx (int): Stopping index for data samples
        img_size (int, optional): Size to resize images to. Defaults to 64.
        device (str, optional): Device to load data onto. Defaults to 'cuda'.
    
    Attributes:
        classes (list): List of class names ['cubes', 'spheres']
        rgb (list): List of preprocessed RGB tensors
        lidar (list): List of preprocessed LiDAR tensors
        class_idxs (list): List of class index tensors
    """
    
    def __init__(self, root_dir, start_idx, stop_idx, img_size=64, device='cuda'):
        self.classes = ["cubes", "spheres"]
        self.root_dir = root_dir
        self.img_size = img_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Image transformations
        self.transforms = transforms.Compose([
            transforms.Resize(img_size),
            # transforms.ToImage(),                         # Currently not compatible with my Environment
            # transforms.ConvertImageDtype(torch.float32),
            transforms.ToTensor(),
        ])
        
        # Storage for preprocessed data
        self.rgb = []
        self.lidar = []
        self.class_idxs = []
        
        # Load data
        self._load_data(start_idx, stop_idx)
    
    def _load_data(self, start_idx, stop_idx):
        """Load and preprocess RGB and LiDAR data from disk."""
        for class_idx, class_name in enumerate(self.classes):
            for idx in range(start_idx, stop_idx):
                file_number = "{:04d}".format(idx)
                
                # Load RGB image
                rgb_path = f"{self.root_dir}{class_name}/rgb/{file_number}.png"
                rgb_img = Image.open(rgb_path)
                rgb_img = self.transforms(rgb_img).to(self.device)
                self.rgb.append(rgb_img)
                
                # Load LiDAR depth map
                lidar_path = f"{self.root_dir}{class_name}/lidar/{file_number}.npy"
                lidar_depth = np.load(lidar_path)
                lidar_depth = torch.from_numpy(lidar_depth[None, :, :]).to(torch.float32).to(self.device)
                self.lidar.append(lidar_depth)
                
                # Store class label
                class_tensor = torch.tensor(class_idx, dtype=torch.float32)[None].to(self.device)
                self.class_idxs.append(class_tensor)
    
    def __len__(self):
        """Return the total number of samples."""
        return len(self.class_idxs)
    
    def __getitem__(self, idx):
        """
        Get a single sample by index.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (rgb_tensor, lidar_tensor, class_label)
        """
        return self.rgb[idx], self.lidar[idx], self.class_idxs[idx]
    
    def get_split_sizes(self, train_batches, batch_size):
        """
        Calculate train/validation split sizes.
        
        Args:
            train_batches (int): Number of batches to reserve for validation
            batch_size (int): Batch size for data loaders
            
        Returns:
            tuple: (train_size, valid_size) per class
        """
        valid_size = train_batches * batch_size
        total_size = len(self) // 2  # Divide by 2 for number per class
        train_size = total_size - valid_size
        return train_size, valid_size
