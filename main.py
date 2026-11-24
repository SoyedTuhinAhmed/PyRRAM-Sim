import argparse
import os
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from pytorchrram import PyRRAMSimulator
from helper import validate, set_seed

# CIFAR-10 trained ResNet18 (94.98% test accuracy, MIT license):
# https://huggingface.co/edadaltocg/resnet18_cifar10
# The hash in the file name pins the SHA-256 of the downloaded checkpoint.
CIFAR10_RESNET18_URL = "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin"
CIFAR10_RESNET18_FILE = "resnet18_cifar10-f6562425.bin"
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def cifar10_resnet18(checkpoint: str = None) -> nn.Module:
    """
    ResNet18 in its CIFAR-10 form (3x3 stride-1 stem, no max-pool, 10 classes) with trained weights.
    Loads `checkpoint` (a state_dict file) if given, otherwise the pretrained weights above.
    """
    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()

    if checkpoint is None:
        state_dict = torch.hub.load_state_dict_from_url(
            CIFAR10_RESNET18_URL, map_location='cpu', check_hash=True,
            file_name=CIFAR10_RESNET18_FILE, weights_only=True,
        )
    else:
        state_dict = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state_dict = state_dict.get('state_dict', state_dict)
        state_dict = {k.removeprefix('module.'): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model

def main():
    """
    Main function to run the RRAM non-ideality simulation from the command line.
    """
    parser = argparse.ArgumentParser(description="PyRRAM-Sim: RRAM Non-Ideality Simulation Toolkit")

    # Model and Dataset Arguments
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18'],
                        help='CIFAR-10 trained model to use for simulation.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Optional state_dict of your own CIFAR-10 ResNet18 (CIFAR stem, trained with the same '
                             'normalization). Defaults to downloading a pretrained checkpoint.')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the CIFAR-10 dataset (downloaded if missing).')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for validation dataloader.')

    # RRAM Hardware Arguments
    parser.add_argument('--g-min', type=float, default=1e-6, help='Minimum RRAM conductance (Siemens).')
    parser.add_argument('--g-max', type=float, default=100e-6, help='Maximum RRAM conductance (Siemens).')
    parser.add_argument('--no-differential', action='store_true', help='Use single-ended RRAM cells instead of differential pairs.')
    parser.add_argument('--diff-style', type=str, default='two_sided', choices=['two_sided', 'one_sided'],
                        help='Differential mapping: constant-sum pair (two_sided) or idle cell at G_min (one_sided).')
    parser.add_argument('--n-levels', type=int, default=None, help='Conductance levels per cell (e.g. 256 for 8-bit, 2 for binary).')
    parser.add_argument('--cells-per-weight', type=int, default=1, help='Cells per weight per polarity (bit-slicing).')

    # Non-Ideality Control Arguments
    parser.add_argument('--apply-stuck-at', action='store_true', help='Flag to inject stuck-at faults before programming.')
    parser.add_argument('--p-on', type=float, default=0.0, help='Probability of a cell stuck at G_max (LRS).')
    parser.add_argument('--p-off', type=float, default=0.0, help='Probability of a cell stuck at G_min (HRS).')
    parser.add_argument('--p-int', type=float, default=0.0, help='Probability of a cell stuck at an intermediate state.')
    parser.add_argument('--compensate', action='store_true', help='Reprogram the healthy partner of a stuck differential cell.')
    parser.add_argument('--fault-seed', type=int, default=None, help='Dedicated seed for the stuck-at fault map.')

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
    print(f"Loading CIFAR-10 trained {args.model} model...")
    model = cifar10_resnet18(args.checkpoint)
    model.to(device)
    model.eval()

    # Load Dataset (CIFAR-10 test set at its native 32x32 resolution)
    print(f"Loading dataset from {args.dataset_path}...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
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
    baseline_accuracy = validate(model, testloader, criterion, device)[-1]
    print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")

    # Initialize Simulator
    print("\n--- Initializing PyRRAMSimulator ---")
    rram_sim = PyRRAMSimulator(
        pretrained_model=model,
        G_min=args.g_min,
        G_max=args.g_max,
        differential=not args.no_differential,
        diff_style=args.diff_style,
        n_levels=args.n_levels,
        cells_per_weight=args.cells_per_weight,
    )

    # Configure and Run Simulation
    print("\n--- Running RRAM Simulation ---")
    
    # Build parameter dictionaries based on flags
    stuck_at_params = None
    if args.apply_stuck_at:
        stuck_at_params = {
            'p_on': args.p_on,
            'p_off': args.p_off,
            'p_int': args.p_int,
            'compensate': args.compensate,
            'fault_seed': args.fault_seed,
        }
        print(f"Applying Stuck-at Faults with params: {stuck_at_params}")

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
        stuck_at_params=stuck_at_params,
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

