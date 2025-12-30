import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbedderMaxPool(nn.Module):
    """
    Convolutional embedder using MaxPool2d for spatial downsampling.
    
    Args:
        in_ch (int): Number of input channels (4 for RGB, 1 for LiDAR)
        emb_size (int, optional): Size of output embedding. Defaults to 100.
    
    Architecture:
        Conv(3x3) + ReLU + MaxPool(2x2) -> [50 channels]
        Conv(3x3) + ReLU + MaxPool(2x2) -> [100 channels]
        Conv(3x3) + ReLU + MaxPool(2x2) -> [200 channels]
        Conv(3x3) + ReLU + MaxPool(2x2) -> [200 channels]
        Flatten -> Dense -> Dense -> Normalized embedding
    """
    
    def __init__(self, in_ch, emb_size=100):
        super().__init__()
        kernel_size = 3
        padding = 1

        # Convolutional layers with MaxPool2d
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 50, kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(50, 100, kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(100, 200, kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(200, 200, kernel_size, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        # Dense embedding layers
        self.dense_emb = nn.Sequential(
            nn.Linear(200 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, emb_size)
        )

    def forward(self, x):
        """
        Forward pass through embedder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Normalized embedding of shape (B, emb_size)
        """
        conv = self.conv(x)
        emb = self.dense_emb(conv)
        return F.normalize(emb)


class EmbedderStrided(nn.Module):
    """
    Convolutional embedder using strided convolutions for downsampling.
    
    Args:
        in_ch (int): Number of input channels (4 for RGB, 1 for LiDAR)
        emb_size (int, optional): Size of output embedding. Defaults to 100.
    
    Architecture:
        Conv(3x3, stride=2) + ReLU -> [50 channels]
        Conv(3x3, stride=2) + ReLU -> [100 channels]
        Conv(3x3, stride=2) + ReLU -> [200 channels]
        Conv(3x3, stride=2) + ReLU -> [200 channels]
        Flatten -> Dense -> Dense -> Normalized embedding
    """
    
    def __init__(self, in_ch, emb_size=100):
        super().__init__()
        kernel_size = 3
        padding = 1

        # Convolutional layers with stride=2 (replaces MaxPool2d)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 50, kernel_size, stride=2, padding=padding),
            nn.ReLU(),
            nn.Conv2d(50, 100, kernel_size, stride=2, padding=padding),
            nn.ReLU(),
            nn.Conv2d(100, 200, kernel_size, stride=2, padding=padding),
            nn.ReLU(),
            nn.Conv2d(200, 200, kernel_size, stride=2, padding=padding),
            nn.ReLU(),
            nn.Flatten()
        )

        # Dense embedding layers (same as MaxPool version)
        self.dense_emb = nn.Sequential(
            nn.Linear(200 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, emb_size)
        )

    def forward(self, x):
        """
        Forward pass through embedder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Normalized embedding of shape (B, emb_size)
        """
        conv = self.conv(x)
        emb = self.dense_emb(conv)
        return F.normalize(emb)


class EarlyFusion(nn.Module):
    """
    Early fusion architecture for RGB-LiDAR classification.
    
    Args:
        rgb_channels (int, optional): Number of RGB channels. Defaults to 4.
        lidar_channels (int, optional): Number of LiDAR channels. Defaults to 1.
        emb_size (int, optional): Size of embeddings. Defaults to 100.
        use_strided (bool, optional): Use strided conv embedder. Defaults to False.
    """
    
    def __init__(self, rgb_channels=4, lidar_channels=1, emb_size=100, use_strided=False):
        super().__init__()
        
        total_channels = rgb_channels + lidar_channels
        
        if use_strided:
            self.conv = nn.Sequential(
                nn.Conv2d(total_channels, 50, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(50, 100, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(100, 200, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(200, 200, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(total_channels, 50, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(50, 100, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(100, 200, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(200, 200, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten()
            )
        
        # Dense embedding layers
        self.embed = nn.Sequential(
            nn.Linear(200 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, emb_size)
        )
        
        # Classifier head
        self.classifier = nn.Linear(emb_size, 1)
    
    def forward(self, rgb, lidar):
        """
        Forward pass through early fusion network.
        
        Args:
            rgb (torch.Tensor): RGB input of shape (B, 4, H, W)
            lidar (torch.Tensor): LiDAR input of shape (B, 1, H, W)
            
        Returns:
            torch.Tensor: Classification logits of shape (B, 1)
        """
        # Concatenate at input level (early fusion)
        x = torch.cat([rgb, lidar], dim=1)  # (B, 5, H, W)
        
        # Convolutional layers
        x = self.conv(x)  # (B, 200*4*4) flattened
        
        # Embedding with normalization
        emb = self.embed(x)  # (B, emb_size)
        emb = F.normalize(emb)
        
        # Classify
        output = self.classifier(emb)  # (B, 1)
        return output


class LateFusion(nn.Module):
    """
    Late fusion architecture for RGB-LiDAR classification.
    
    Args:
        rgb_channels (int, optional): Number of RGB channels. Defaults to 4.
        lidar_channels (int, optional): Number of LiDAR channels. Defaults to 1.
        emb_size (int, optional): Size of embeddings. Defaults to 100.
        use_strided (bool, optional): Use strided conv embedder. Defaults to False.
    """
    
    def __init__(self, rgb_channels=4, lidar_channels=1, emb_size=100, use_strided=False):
        super().__init__()
        
        # Choose embedder type
        embedder_class = EmbedderStrided if use_strided else EmbedderMaxPool
        
        # RGB and LiDAR encoders
        self.rgb_encoder = embedder_class(rgb_channels, emb_size)
        self.lidar_encoder = embedder_class(lidar_channels, emb_size)
        
        # Classifier head (takes concatenated embeddings)
        self.classifier = nn.Sequential(
            nn.Linear(emb_size * 2, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
    
    def forward(self, rgb, lidar):
        """
        Forward pass through late fusion network.
        
        Args:
            rgb (torch.Tensor): RGB input of shape (B, 4, H, W)
            lidar (torch.Tensor): LiDAR input of shape (B, 1, H, W)
            
        Returns:
            torch.Tensor: Classification logits of shape (B, 1)
        """
        rgb_emb = self.rgb_encoder(rgb)
        lidar_emb = self.lidar_encoder(lidar)
        
        # Concatenate embeddings
        combined = torch.cat([rgb_emb, lidar_emb], dim=1)
        
        # Classify
        output = self.classifier(combined)
        return output


class IntermediateFusion(nn.Module):
    """
    Intermediate fusion architecture for RGB-LiDAR classification.
    
    Args:
        rgb_channels (int, optional): Number of RGB channels. Defaults to 4.
        lidar_channels (int, optional): Number of LiDAR channels. Defaults to 1.
        merge_method (str, optional): Fusion method ('concat', 'add', 'hadamard'). 
                                     Defaults to 'concat'.
        use_strided (bool, optional): Use strided conv instead of MaxPool2d. Defaults to False.
    """
    
    def __init__(self, rgb_channels=4, lidar_channels=1, merge_method='concat', use_strided=False):
        super().__init__()
        self.merge_method = merge_method
        
        # Early RGB convolutions
        if use_strided:
            self.rgb_conv1 = nn.Sequential(
                nn.Conv2d(rgb_channels, 50, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )
            self.rgb_conv2 = nn.Sequential(
                nn.Conv2d(50, 100, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )
        else:
            self.rgb_conv1 = nn.Sequential(
                nn.Conv2d(rgb_channels, 50, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.rgb_conv2 = nn.Sequential(
                nn.Conv2d(50, 100, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        
        # Early LiDAR convolutions
        if use_strided:
            self.lidar_conv1 = nn.Sequential(
                nn.Conv2d(lidar_channels, 50, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )
            self.lidar_conv2 = nn.Sequential(
                nn.Conv2d(50, 100, kernel_size=3, stride=2, padding=1),
                nn.ReLU()
            )
        else:
            self.lidar_conv1 = nn.Sequential(
                nn.Conv2d(lidar_channels, 50, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.lidar_conv2 = nn.Sequential(
                nn.Conv2d(50, 100, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
        
        # Shared layers after fusion
        # For concatenation, input channels = 200 (100 + 100)
        # For addition/hadamard, input channels = 100
        in_channels = 200 if merge_method == 'concat' else 100
        
        if use_strided:
            self.shared_conv = nn.Sequential(
                nn.Conv2d(in_channels, 200, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(200, 200, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten()
            )
        else:
            self.shared_conv = nn.Sequential(
                nn.Conv2d(in_channels, 200, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(200, 200, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten()
            )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(200 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
    
    def forward(self, rgb, lidar):
        """
        Forward pass through intermediate fusion network.
        
        Args:
            rgb (torch.Tensor): RGB input of shape (B, 4, H, W)
            lidar (torch.Tensor): LiDAR input of shape (B, 1, H, W)
            
        Returns:
            torch.Tensor: Classification logits of shape (B, 1)
        """
        # Process through early layers
        rgb_feat = self.rgb_conv1(rgb)
        rgb_feat = self.rgb_conv2(rgb_feat)
        
        lidar_feat = self.lidar_conv1(lidar)
        lidar_feat = self.lidar_conv2(lidar_feat)
        
        # Merge feature maps based on method
        if self.merge_method == 'concat':
            merged = torch.cat([rgb_feat, lidar_feat], dim=1)
        elif self.merge_method == 'add':
            merged = rgb_feat + lidar_feat
        elif self.merge_method == 'hadamard':
            merged = rgb_feat * lidar_feat
        else:
            raise ValueError(f"Unknown merge method: {self.merge_method}")
        
        # Process through shared layers
        features = self.shared_conv(merged)
        
        # Classify
        output = self.classifier(features)
        return output


def count_parameters(model):
    """
    Count the total number of trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Total number of trainable parameters
        
    Example:
        >>> model = LateFusion()
        >>> num_params = count_parameters(model)
        >>> print(f"Model has {num_params:,} parameters")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
