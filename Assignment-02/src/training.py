import time
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import wandb
import os


def seed_everything(seed=42):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed (int, optional): Random seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def seed_worker(worker_id):
    """
    Seed function for DataLoader workers to ensure reproducibility.
    
    Args:
        worker_id (int): Worker ID provided by DataLoader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_predictions(outputs):
    """
    Convert logits to binary predictions.
    
    Args:
        outputs (torch.Tensor): Model output logits
        
    Returns:
        torch.Tensor: Binary predictions (0 or 1)
    """
    return (torch.sigmoid(outputs) > 0.5).float()


def calculate_f1(outputs, targets):
    """
    Calculate F1 score for binary classification.
    
    Args:
        outputs (torch.Tensor): Model output logits
        targets (torch.Tensor): Ground truth labels
        
    Returns:
        float: F1 score (0-1)
    """
    preds = get_predictions(outputs).cpu().numpy()
    targets = targets.cpu().numpy()
    return f1_score(targets, preds, average='binary')


def calculate_accuracy(outputs, targets):
    """
    Calculate accuracy for binary classification.
    
    Args:
        outputs (torch.Tensor): Model output logits
        targets (torch.Tensor): Ground truth labels
        
    Returns:
        float: Accuracy (0-1)
    """
    preds = get_predictions(outputs)
    correct = (preds == targets).float().sum()
    return (correct / len(targets)).item()


def get_model_memory_usage():
    """
    Get current GPU memory usage.
    
    Returns:
        float: Memory usage in MB, or 0 if CUDA not available
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


def train_model(model, model_name, train_dataloader, valid_dataloader,
                train_N, valid_N, subset_size, batch_size,
                epochs=20, lr=0.001, wandb_project="cilp-extended-assessment",
                wandb_group="training", wandb_tags=None, seed=42,
                task_type="fusion", architecture_type=None, pooling_type=None,
                track_memory=True):
    """    
    Args:
        model (nn.Module): Model to train
        model_name (str): Name for W&B run
        train_dataloader (DataLoader): Training data loader
        valid_dataloader (DataLoader): Validation data loader
        train_N (int): Total training samples
        valid_N (int): Total validation samples
        subset_size (int): Subset size per class
        batch_size (int): Batch size
        epochs (int, optional): Number of training epochs. Defaults to 20.
        lr (float, optional): Learning rate. Defaults to 0.001.
        wandb_project (str, optional): W&B project name
        wandb_group (str, optional): W&B group name
        wandb_tags (list, optional): Additional W&B tags
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        task_type (str, optional): Type of task ("fusion" or "ablation"). Defaults to "fusion".
        architecture_type (str, optional): Architecture type for tracking. Defaults to None.
        pooling_type (str, optional): Pooling type for tracking. Defaults to None.
        track_memory (bool, optional): Whether to track GPU memory. Defaults to True.
    
    Returns:
        dict: Dictionary with metrics (varies by task_type):
            - fusion: val_loss, val_accuracy, val_f1, parameters, time_per_epoch, memory_mb
            - ablation: adds architecture and pooling_type fields
    """
    from .models import count_parameters
    
    # Set seed for reproducibility
    seed_everything(seed)
    
    if wandb_tags is None:
        wandb_tags = []
    
    # Configure tags based on task type
    if task_type == "fusion":
        base_tags = ["fusion", "task3", f"subset_{subset_size}"]
    elif task_type == "ablation":
        base_tags = ["ablation", "task4", "maxpool-vs-strided"]
    else:
        base_tags = ["training"]
    
    # Initialize W&B run
    config = {
        "model": model_name,
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "subset_size": subset_size,
        "train_samples": train_N,
        "valid_samples": valid_N,
        "parameters": count_parameters(model),
        "task_type": task_type
    }
    
    # Add optional config fields
    if architecture_type:
        config["architecture"] = architecture_type
    if pooling_type:
        config["pooling"] = pooling_type
    
    run = wandb.init(
        project=wandb_project,
        group=wandb_group,
        name=model_name,
        tags=base_tags + wandb_tags,
        config=config
    )
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Track metrics
    best_val_loss = float('inf')
    best_val_f1 = 0
    best_val_acc = 0
    epoch_times = []
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    if architecture_type:
        print(f"Architecture: {architecture_type}")
    if pooling_type:
        print(f"Pooling: {pooling_type}")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        for rgb, lidar, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(rgb, lidar)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.append(outputs.detach())
            train_targets.append(labels)
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_dataloader)
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_f1 = calculate_f1(train_preds, train_targets)
        train_acc = calculate_accuracy(train_preds, train_targets)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for rgb, lidar, labels in valid_dataloader:
                outputs = model(rgb, lidar)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_preds.append(outputs)
                val_targets.append(labels)
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(valid_dataloader)
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_f1 = calculate_f1(val_preds, val_targets)
        val_acc = calculate_accuracy(val_preds, val_targets)
        
        # Track best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_f1 = val_f1
            best_val_acc = val_acc
        
        # Timing
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Prepare W&B log
        log_dict = {
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "train/f1": train_f1,
            "train/accuracy": train_acc,
            "valid/loss": avg_val_loss,
            "valid/f1": val_f1,
            "valid/accuracy": val_acc,
            "time/epoch_seconds": epoch_time
        }
        
        # Add memory tracking if enabled
        if track_memory:
            memory_mb = get_model_memory_usage()
            log_dict["memory/gpu_mb"] = memory_mb
        
        wandb.log(log_dict)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
                  f"Time: {epoch_time:.2f}s")
    
    # Final metrics
    avg_epoch_time = np.mean(epoch_times)
    final_memory = get_model_memory_usage() if track_memory else 0.0
    
    # Log final summary
    summary = {
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_acc,
        "best_val_f1": best_val_f1,
        "avg_epoch_time": avg_epoch_time
    }
    if track_memory:
        summary["final_memory_mb"] = final_memory
    
    wandb.summary.update(summary)
    wandb.finish()
    
    # Return metrics for comparison table
    result = {
        "model": model_name,
        "val_loss": best_val_loss,
        "val_accuracy": best_val_acc,
        "val_f1": best_val_f1,
        "parameters": count_parameters(model),
        "time_per_epoch": avg_epoch_time,
        "memory_mb": final_memory
    }
    
    # Add architecture/pooling info for ablation studies
    if architecture_type:
        result["architecture"] = architecture_type
    if pooling_type:
        result["pooling_type"] = pooling_type
    
    return result


# Backward compatibility wrappers
def train_fusion_model(model, model_name, train_dataloader, valid_dataloader,
                       train_N, valid_N, subset_size, batch_size,
                       epochs=20, lr=0.001, wandb_project="cilp-extended-assessment",
                       wandb_group="fusion-comparison", wandb_tags=None, seed=42):
    """
    Train a fusion model with W&B logging for Task 3.
    
    This is a wrapper around train_model() for backward compatibility.
    
    Args:
        model (nn.Module): Fusion model to train
        model_name (str): Name for W&B run
        train_dataloader (DataLoader): Training data loader
        valid_dataloader (DataLoader): Validation data loader
        train_N (int): Total training samples
        valid_N (int): Total validation samples
        subset_size (int): Subset size per class
        batch_size (int): Batch size
        epochs (int, optional): Number of training epochs. Defaults to 20.
        lr (float, optional): Learning rate. Defaults to 0.001.
        wandb_project (str, optional): W&B project name
        wandb_group (str, optional): W&B group name
        wandb_tags (list, optional): Additional W&B tags
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    
    Returns:
        dict: Dictionary with final metrics (val_loss, val_f1, parameters, 
              time_per_epoch, memory_mb)
    """
    return train_model(
        model, model_name, train_dataloader, valid_dataloader,
        train_N, valid_N, subset_size, batch_size,
        epochs=epochs, lr=lr, wandb_project=wandb_project,
        wandb_group=wandb_group, wandb_tags=wandb_tags, seed=seed,
        task_type="fusion", track_memory=True
    )


def train_ablation_model(model, model_name, train_dataloader, valid_dataloader,
                         train_N, valid_N, subset_size, batch_size,
                         epochs=20, lr=0.001, wandb_project="cilp-extended-assessment",
                         wandb_group="ablation-maxpool-vs-strided", wandb_tags=None, seed=42,
                         architecture_type="Unknown", pooling_type="Unknown"):
    """
    Train a model for ablation study with W&B logging (Task 4).
    
    Args:
        model (nn.Module): Model to train
        model_name (str): Name for W&B run
        train_dataloader (DataLoader): Training data loader
        valid_dataloader (DataLoader): Validation data loader
        train_N (int): Total training samples
        valid_N (int): Total validation samples
        subset_size (int): Subset size per class
        batch_size (int): Batch size
        epochs (int, optional): Number of training epochs. Defaults to 20.
        lr (float, optional): Learning rate. Defaults to 0.001.
        wandb_project (str, optional): W&B project name
        wandb_group (str, optional): W&B group name
        wandb_tags (list, optional): Additional W&B tags
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        architecture_type (str, optional): Architecture type for tracking. Defaults to "Unknown".
        pooling_type (str, optional): Pooling type for tracking. Defaults to "Unknown".
    
    Returns:
        dict: Dictionary with final metrics (architecture, pooling_type, val_loss,
              val_accuracy, val_f1, parameters, time_per_epoch, memory_mb)
    """
    return train_model(
        model, model_name, train_dataloader, valid_dataloader,
        train_N, valid_N, subset_size, batch_size,
        epochs=epochs, lr=lr, wandb_project=wandb_project,
        wandb_group=wandb_group, wandb_tags=wandb_tags, seed=seed,
        task_type="ablation", architecture_type=architecture_type,
        pooling_type=pooling_type, track_memory=True
    )
