import torch, random, numpy as np

def set_seed(seed):
    torch.use_deterministic_algorithms(True)  # Ensures deterministic behavior
    torch.manual_seed(seed)  # Seed for PyTorch (CPU)
    torch.cuda.manual_seed(seed)  # Seed for PyTorch (single GPU)
    torch.cuda.manual_seed_all(seed)  # Seed for PyTorch (all GPUs, if applicable)
    np.random.seed(seed)  # Seed for NumPy
    random.seed(seed)  # Seed for Python random
    # For compatibility with older PyTorch versions:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.inference_mode()
def validate(model: torch.nn.Module,
             val_loader: torch.utils.data.DataLoader,
             criterion: torch.nn.Module,
             device: torch.device) -> tuple[float, float]:
    """
    Evaluates the model on the validation set.
    Returns (average validation loss, validation accuracy in %).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * correct / total_samples
    return avg_loss, accuracy
