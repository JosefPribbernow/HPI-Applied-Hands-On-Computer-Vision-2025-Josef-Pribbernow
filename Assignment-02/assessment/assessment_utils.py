import torch
import torch.nn as nn
import wandb

# Please do no alter this file. That will make it harder to pass the assessment!
class Classifier(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        kernel_size = 3
        n_classes = 1
        self.embedder = nn.Sequential(
            nn.Conv2d(in_ch, 50, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(50, 100, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(100, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(200, 200, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(200 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, n_classes)
        )

    def get_embs(self, imgs):
        return self.embedder(imgs)
    
    def forward(self, raw_data=None, data_embs=None):
        assert (raw_data is not None or data_embs is not None), "No images or embeddings given."
        if raw_data is not None:
            data_embs = self.get_embs(raw_data)
        return self.classifier(data_embs)


def print_CILP_results(epoch, loss, logits_per_img, is_train=True):
    if is_train:
        print(f"Epoch {epoch}")
        print(f"Train Loss: {loss} ")
    else:
        print(f"Valid Loss: {loss} ")
    print("Similarity:")
    print(logits_per_img)


def print_loss(epoch, loss, is_train=True, is_debug=False):
    loss_type = "Train" if is_train else "Valid"
    out_string = f"Epoch {epoch:3d} | {loss_type} Loss: {loss:2.4f}"
    print(out_string)

def train_model(model, optimizer, loss_func, epochs, train_dataloader, valid_dataloader, 
                wandb_project=None, wandb_name=None, wandb_config=None):
    # Initialize W&B if project is provided
    if wandb_project:
        # Extract group and tags from config if provided
        group = wandb_config.pop("group", None) if wandb_config else None
        tags = wandb_config.pop("tags", None) if wandb_config else None
        
        config = {
            "learning_rate": optimizer.param_groups[0]['lr'],
            "architecture": wandb_config.get("architecture", "Model") if wandb_config else "Model",
            "batch_size": train_dataloader.batch_size,
            "epochs": epochs,
            "optimizer": optimizer.__class__.__name__,
            "scheduler": "ReduceLROnPlateau",
            "num_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        # Merge with any additional config provided
        if wandb_config:
            config.update(wandb_config)
        
        wandb.init(
            project=wandb_project,
            group=group,
            name=wandb_name or "training_run",
            tags=tags,
            config=config
        )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = loss_func(model, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss = train_loss / (step + 1)
        print_loss(epoch, train_loss, is_train=True)
        
        model.eval()
        valid_loss = 0
        for step, batch in enumerate(valid_dataloader):
            loss = loss_func(model, batch)
            valid_loss += loss.item()
        valid_loss = valid_loss / (step + 1)
        print_loss(epoch, valid_loss, is_train=False)
        
        # Step the scheduler based on validation loss
        scheduler.step(valid_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to W&B if initialized
        if wandb_project and wandb.run:
            wandb.log({
                "train/loss": train_loss,
                "valid/loss": valid_loss,
                "learning_rate": current_lr,
                "epoch": epoch,
            })
    
    # Finish W&B run if it was initialized
    if wandb_project and wandb.run:
        wandb.finish()