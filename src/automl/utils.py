from typing import Any
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import TrivialAugmentWide
from torchvision.transforms import RandAugment

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


def get_default_transforms(meta, resized_res=None):
    """
    Returns a torchvision transformation pipeline that resizes and normalizes images
    without applying any data augmentation.

    Args:
        meta (dict): Contains 'image_resolution' and 'num_channels'.
        resized_res (tuple or None): Optional (H, W) tuple to override the default resolution.

    Returns:
        transform (torchvision.transforms.Compose): A composed transform with resizing,
        tensor conversion, and normalization.
    """

    image_size = resized_res if resized_res is not None else meta["image_resolution"]
    num_channels = meta["num_channels"]

    if resized_res is not None:
        print(f"get_default_transforms image reso is downsized from {meta['image_resolution']} to {resized_res}")
    else:
        print(f"get_default_transforms image reso is not downsized {meta['image_resolution']}")
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


def get_augmented_transforms(meta, resized_res=None, augment_type="trivial"):
    """
    Returns a transform pipeline that includes TrivialAugmentWide for data augmentation.

    Args:
        meta (dict): Contains 'image_resolution' and 'num_channels'.
        resized_res (tuple or None): Optional (H, W) tuple to override the default resolution.

    Returns:
        transform (torchvision.transforms.Compose): A composed transform with resizing,
        TrivialAugmentWide, tensor conversion, and normalization.
    """

    image_size = resized_res if resized_res is not None else meta["image_resolution"]

    num_channels = meta["num_channels"]
    if resized_res is not None:
        print(f"get_augmented_transforms image reso is downsized from {meta['image_resolution']} to {resized_res}")    
    else:
        print(f"get_augmented_transforms image reso is not downsized {meta['image_resolution']}")
    if num_channels == 1:
        normalize = transforms.Normalize((0.5,), (0.5,))
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    augment_ops = [transforms.Resize(image_size)]

    if augment_type == "trivial":
        from torchvision.transforms import TrivialAugmentWide
        augment_ops.append(TrivialAugmentWide(num_magnitude_bins=31))
    elif augment_type == "rand":
        from torchvision.transforms import RandAugment
        augment_ops.append(RandAugment())
    elif augment_type == "none":
        pass  # no augmentation

    augment_ops.extend([
        transforms.ToTensor(),
        normalize
    ])

    return transforms.Compose(augment_ops)

