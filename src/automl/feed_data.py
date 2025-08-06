import os
from PIL import Image
from torch.utils.data import Dataset
from utils import get_default_transforms, get_augmented_transforms
from data_analyze import get_undersampled_classes
from collections import Counter

class BalancedDataset(Dataset):
    def __init__(self, dataframe, images_path, metadata, resized_res=None, min_samples_per_class=None):
        """
        Creates a dataset that duplicates samples from undersampled classes
        and applies on-the-fly augmentation only to those duplicated samples.

        Args:
            dataframe (pd.DataFrame): ['image_file_name', 'label']
            images_path (str): Folder where image files are located
            metadata (dict): Contains image size, imbalance flag, and undersampling info
        """
        self.df = dataframe
        self.images_path = images_path
        self.metadata = metadata
        self.resized_res = resized_res

        # Figure out which classes are undersampled
        class_counts = dict(Counter(dataframe['label']))
        if min_samples_per_class is not None:
            # HPO mode: recalculate undersampled_classes based on selected threshold
            self.undersampled_classes = {
                str(cls): {
                    "current_count": count,
                    "target_count": min_samples_per_class,
                    "augmentation_multiplier": round(min_samples_per_class / count, 2)
                }
                for cls, count in class_counts.items()
                if count < min_samples_per_class
            }
        else:
            # NAS mode: use precomputed undersampled_classes from metadata
            self.undersampled_classes = metadata.get("undersampled_classes", {})

        print(f"[DEBUG] min_samples_per_class used in dataset build: {min_samples_per_class}")
        print(f"[DEBUG] undersampled_classes keys: {list(self.undersampled_classes.keys())}")

        # Read original image resolution from metadata or resized
        self.resolution = resized_res if resized_res is not None else metadata["image_resolution"]
        self.dataset_is_imbalanced = metadata.get("is_imbalanced", False)
        self.applied_ops_logger = set()
        self.data = []
        self.duplication_log = {}

        # --- Duplication Loop ---
        for _, row in self.df.iterrows():
            label = str(row['label'])
            img_path = os.path.join(images_path, row['image_file_name'])
            self.data.append((img_path, label, False))  # always keep the original

            if label in self.undersampled_classes:
                if min_samples_per_class is not None:
                    # Use fixed target count for HPO
                    target_count = min_samples_per_class
                    repeat_factor = max(int(target_count / self.undersampled_classes[label]['current_count']) - 1, 0)
                else:
                    # Use multiplier from NAS metadata
                    multiplier = self.undersampled_classes[label]['augmentation_multiplier']
                    repeat_factor = max(int(multiplier - 1), 0)

                if repeat_factor > 0:
                    self.data.extend([(img_path, label, True)] * repeat_factor)
                    self.duplication_log[label] = self.duplication_log.get(label, 0) + repeat_factor

        # --- Post-augmentation logging ---
        aug_counts = dict(Counter([int(lbl) for _, lbl, _ in self.data]))
        print(f"[DEBUG] Final class distribution after augmentation: {aug_counts}")

        with open("class_distribution.log", "a") as f:
            f.write(f"\n--- AFTER augmentation (min_samples_per_class={min_samples_per_class}) ---\n")
            for cls, cnt in sorted(aug_counts.items()):
                f.write(f"  Class {cls}: {cnt} samples\n")

        # Define transforms
        self.default_transform = get_default_transforms(self.metadata, resized_res)
        self.trivial_transform = get_augmented_transforms(self.metadata, resized_res)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label, duplicated = self.data[idx]
        image = Image.open(img_path)
        if self.metadata.get("is_grayscale", False):
            image = image.convert("L")
        else:
            image = image.convert("RGB")

        if duplicated:
            image = self.trivial_transform(image)
        else:
            image = self.default_transform(image)

        if idx == 0:
            print(f"[DEBUG] Shape of image[{idx}]: {image.shape}")
        return image, int(label)
