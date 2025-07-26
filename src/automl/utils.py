from typing import Any
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import TrivialAugmentWide


def calculate_mean_std(dataset_class: Any):
    """Calculate the mean and standard deviation of the entire image dataset."""
    mean = 0.
    std = 0.
    total_images_count = 0

    dataset = dataset_class(
        root="./data",
        split='train',
        download=True,
        transform=transforms.ToTensor()
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    for images, _ in loader:
        batch_samples = images.size(0)  # last batch might be smaller
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std


def get_default_transforms(meta):
    """
    Returns a torchvision transformation pipeline based on dataset metadata.
    Resizes and normalizes only (no augmentation).
    
    Args:
        meta (dict): Contains 'image_resolution' and 'num_channels'.
    
    Returns:
        transform: torchvision.transforms.Compose
    """
    image_size = meta["image_resolution"]
    num_channels = meta["num_channels"]
    print(f"Image size found in the dataset is: {image_size}")
    if num_channels == 1:
        normalize = transforms.Normalize((0.5,), (0.5,))
    else:
        print(f"num channels > 1 → {num_channels}")
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalize
    ])


def get_augmented_transforms(meta):
    """
    Returns a transform pipeline that includes TrivialAugmentWide for data augmentation.
    
    Args:
        meta (dict): Contains 'image_resolution' and 'num_channels'.
    
    Returns:
        transform: torchvision.transforms.Compose
    """
    image_size = meta["image_resolution"]
    num_channels = meta["num_channels"]

    if num_channels == 1:
        normalize = transforms.Normalize((0.5,), (0.5,))
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize(image_size),
        TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
        normalize
    ])
