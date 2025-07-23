import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, vgg16, mobilenet_v2
from pytorchrram import PyRRAMSimulator
from helper import validate, set_seed

def main():
    """
    Main function to run the RRAM non-ideality simulation from the command line.
    """
    parser = argparse.ArgumentParser(description="PyRRAM-Sim: RRAM Non-Ideality Simulation Toolkit")

    # Model and Dataset Arguments
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'vgg16', 'mobilenet_v2'],
                        help='Pre-trained model to use for simulation.')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset (e.g., CIFAR-10).')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for validation dataloader.')

    # RRAM Hardware Arguments
    parser.add_argument('--g-min', type=float, default=1e-6, help='Minimum RRAM conductance (Siemens).')
    parser.add_argument('--g-max', type=float, default=100e-6, help='Maximum RRAM conductance (Siemens).')
    parser.add_argument('--no-differential', action='store_true', help='Use single-ended RRAM cells instead of differential pairs.')

    # Non-Ideality Control Arguments
    parser.add_argument('--apply-prog-error', action='store_true', help='Flag to apply programming error.')
    parser.add_argument('--prog-alpha-ind', type=float, default=0.03, help='State-independent noise factor for programming error.')
    parser.add_argument('--prog-alpha-prop', type=float, default=0.0, help='State-proportional noise factor for programming error.')

    parser.add_argument('--apply-relaxation', action='store_true', help='Flag to apply conductance relaxation.')
    parser.add_argument('--relax-alpha-ind', type=float, default=0.07, help='State-independent noise factor for relaxation.')
    
    parser.add_argument('--apply-drift', action='store_true', help='Flag to apply relative drift.')
    parser.add_argument('--relative-drift', type=float, default=0.2, help='Relative drift factor (from DoRA paper model).')

    # Simulation Arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    print(f"Loading pre-trained {args.model} model...")
    if args.model == 'resnet18':
        model = resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    elif args.model == 'vgg16':
        model = vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
    elif args.model == 'mobilenet_v2':
        model = mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
    
    model.to(device)
    model.eval()

    # Load Dataset (Example: CIFAR-10)
    print(f"Loading dataset from {args.dataset_path}...")
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    try:
        testset = torchvision.datasets.CIFAR10(root=args.dataset_path, train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
        criterion = nn.CrossEntropyLoss()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset path is correct and the dataset is available.")
        return

    # Run Baseline Validation
    print("\n--- Running Baseline (Ideal Digital Model) ---")
    baseline_accuracy = validate(model, testloader, criterion, device)
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")

    # Initialize Simulator
    print("\n--- Initializing PyRRAMSimulator ---")
    rram_sim = PyRRAMSimulator(
        pretrained_model=model,
        G_min=args.g_min,
        G_max=args.g_max,
        differential=not args.no_differential
    )

    # Configure and Run Simulation
    print("\n--- Running RRAM Simulation ---")
    
    # Build parameter dictionaries based on flags
    prog_params = None
    if args.apply_prog_error:
        prog_params = {
            'alpha_ind': args.prog_alpha_ind,
            'alpha_prop': args.prog_alpha_prop
        }
        print(f"Applying Programming Error with params: {prog_params}")

    relax_params = None
    if args.apply_relaxation:
        relax_params = {'alpha_ind': args.relax_alpha_ind}
        print(f"Applying Relaxation with params: {relax_params}")

    drift_params = None
    if args.apply_drift:
        drift_params = {'relative_drift': args.relative_drift}
        print(f"Applying Relative Drift with params: {drift_params}")

    # Run the full simulation
    rram_accuracy = rram_sim.simulate(
        dataloader=testloader,
        criterion=criterion,
        programming_error_params=prog_params,
        relaxation_params=relax_params,
        drift_params=drift_params
    )

    print("\n--- Simulation Results ---")
    print(f"Baseline Digital Accuracy: {baseline_accuracy:.2f}%")
    print(f"Final RRAM Accuracy: {rram_accuracy:.2f}%")
    print(f"Accuracy Drop: {baseline_accuracy - rram_accuracy:.2f}%")


if __name__ == '__main__':
    main()

