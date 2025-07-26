import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from collections import Counter
import json

import numpy as np
from scipy.stats import skew
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import math

class AutoMLDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None):
        """
        Args:
            csv_path (str): Path to train.csv or test.csv
            images_dir (str): Directory where images are stored
            transform (callable): Torchvision transform to apply to images
        """
        self.df = pd.read_csv(csv_path)
        self.images_dir = os.path.expanduser(images_dir)
        self.transform = transform

        self.labels = self.df['label'].values
        self.image_paths = self.df['image_file_name'].values

        # Infer shape and channels from a sample image
        sample_img_path = os.path.join(self.images_dir, self.image_paths[0])
        with Image.open(sample_img_path) as img:
            self.img_size = img.size[::-1]  # (height, width)
            self.num_channels = len(img.getbands())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_paths[idx])
        with Image.open(img_path) as img:
            img = img.convert("RGB")  # force 3 channels
            if self.transform:
                img = self.transform(img)
        label = int(self.labels[idx])
        return img, label

    def get_class_distribution(self):
        count = Counter(self.labels)
        total = len(self.labels)
        return {str(cls): round(count[cls] / total, 4) for cls in count}

    def get_metadata(self):
        return {
            "num_samples": len(self.df),
            "num_classes": len(set(self.labels)),
            "image_resolution": self.img_size,      # (H, W)
            "num_channels": self.num_channels,      # 1 or 3
            "class_distribution": self.get_class_distribution()
        }

def analyze_dataset(dataset_name: str, save_to_file: bool = True):
    base_path = os.path.expanduser("~/Documents/AutoML/data")
    dataset_path = os.path.join(base_path, dataset_name)

    # Load CSV
    csv_path = os.path.join(dataset_path, "train.csv")
    img_dir = os.path.join(dataset_path, "images_train")

    df = pd.read_csv(csv_path)
    labels = df["label"].tolist()
    image_files = df["image_file_name"].tolist()

    # Basic stats
    num_samples = len(labels)
    num_classes = len(set(labels))
    class_counts = dict(Counter(labels))
    class_freqs = {k: v / num_samples for k, v in class_counts.items()}

    # Compute mean and median frequency
    freq_values = list(class_freqs.values())
    mean_freq = round(float(np.mean(freq_values)), 6)
    median_freq = round(float(np.median(freq_values)), 6)

    # One image to get resolution and channels
    first_img_path = os.path.join(img_dir, image_files[0])
    with Image.open(first_img_path) as img:
        width, height = img.size
        num_channels = len(img.getbands())

    metadata = {
        "dataset": dataset_name,
        "num_samples": num_samples,
        "num_classes": num_classes,
        "image_resolution": (width, height),
        "num_channels": num_channels,
        "class_counts": class_counts,
        "class_distribution": {
            str(k): round(v, 4)
            for k, v in sorted(class_freqs.items(), key=lambda item: int(item[0]))
        },
        "mean_class_frequency": mean_freq,
        "median_class_frequency": median_freq,
        "is_imbalanced": None,  # will be decided below
        "undersampled_classes": {}  # will be filled later
    }

    print("\U0001F4CA Dataset Metadata:")
    #print(json.dumps(metadata, indent=2))

    return metadata

imbalance_votes = []

def compute_entropy(class_probs):
    entropy = -np.sum([p * np.log2(p) for p in class_probs if p > 0])
    max_entropy = np.log2(len(class_probs))
    print(f"[Entropy] Value: {entropy:.4f} / Max: {max_entropy:.4f}")
    result = entropy < 0.75 * max_entropy
    imbalance_votes.append(result)
    print("🔴 Entropy suggests class imbalance" if result else "🟢 Entropy suggests reasonably balanced")

def compute_gini(class_probs):
    gini = 1 - np.sum([p ** 2 for p in class_probs])
    max_gini = 1 - (1 / len(class_probs))
    print(f"[Gini Index] Value: {gini:.4f} / Max: {max_gini:.4f}")
    result = gini < 0.75 * max_gini
    imbalance_votes.append(result)
    print("🔴 Gini suggests class imbalance" if result else "🟢 Gini suggests reasonably balanced")

def compute_skewness(class_probs, total_samples):
    counts = np.array([p * total_samples for p in class_probs])
    skew_val = skew(counts)
    print(f"[Skewness] Value: {skew_val:.4f}")
    result = abs(skew_val) > 1
    imbalance_votes.append(result)
    print("🔴 Skewness suggests imbalance (heavy-tailed)" if result else "🟢 Skewness is acceptable")

def compute_kmeans_clustering(class_probs):
    X = np.array(class_probs).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(X)
    counts = np.bincount(kmeans.labels_)
    ratio = min(counts) / max(counts)
    print(f"[KMeans Clustering] Cluster ratio: {ratio:.4f}")
    result = ratio < 0.3
    imbalance_votes.append(result)
    print("🔴 KMeans found a dominant vs minority class split" if result else "🟢 KMeans found no clear imbalance clusters")

def detect_outliers_isolation_forest(class_probs):
    X = np.array(class_probs).reshape(-1, 1)
    clf = IsolationForest(contamination=0.1, random_state=42).fit(X)
    outlier_flags = clf.predict(X)  # -1: outlier, 1: inlier
    n_outliers = np.sum(outlier_flags == -1)
    print(f"[Isolation Forest] Detected {n_outliers} minority class(es) as outliers")
    result = n_outliers > len(class_probs) * 0.05
    imbalance_votes.append(result)
    print("🔴 Isolation Forest indicates possible imbalance" if result else "🟢 No major outliers detected")

def check_imbalance(metadata):
    print(f"\n📊 Checking imbalance for dataset: {metadata['dataset']}")
    class_distribution = metadata['class_distribution']
    total_samples = metadata['num_samples']
    class_probs = list(class_distribution.values())

    imbalance_votes.clear()
    compute_entropy(class_probs)
    compute_gini(class_probs)
    compute_skewness(class_probs, total_samples)
    compute_kmeans_clustering(class_probs)
    detect_outliers_isolation_forest(class_probs)

    metadata['is_imbalanced'] = imbalance_votes.count(True) >= 3
    print(f"\n📌 Final Decision: {'🔴 Imbalanced' if metadata['is_imbalanced'] else '🟢 Balanced'}")

def calculate_median_margin(class_counts):
    """
    Calculates the median of class sample counts.
    """
    # If class_counts contains dicts, extract 'current_count'
    sample_values = list(class_counts.values())
    for v in sample_values:
        print("🔍 Type:", type(v), "| Value:", v)
    if isinstance(sample_values[0], dict):
        counts = np.array([v["current_count"] for v in sample_values])
    else:
        counts = np.array(sample_values)
    median = np.median(counts)
    return int(median)

def get_undersampled_classes(class_counts, min_samples_per_class):
    """
    Identify undersampled classes:
    - Classes with count < min_samples_per_class
    - OR classes with count < median class size
    """
    median_count = calculate_median_margin(class_counts)
    undersampled = {}

    for cls, count in class_counts.items():
        if count < min_samples_per_class or count < median_count:
            target = max(median_count, min_samples_per_class)
            multiplier = round(target / count, 2) if count > 0 else None
            undersampled[cls] = {
                "current_count": count,
                "target_count": target,
                "augmentation_multiplier": multiplier
            }

    return undersampled



if __name__ == "__main__":
    metadata = analyze_dataset("flowers", save_to_file=False)  # <-- disable early save
    check_imbalance(metadata)

    undersampled_info = get_undersampled_classes(metadata["class_counts"], min_samples_per_class=200)
    metadata["undersampled_classes"] = undersampled_info

    print("🔍 Undersampled Classes Detected:")
    #for cls, info in undersampled_info.items():
       # print(f"Class {cls}: {info['current_count']} ➡ {info['target_count']} (×{info['augmentation_multiplier']})")

    # ✅ Now save the updated version
    # Save it to root of project
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    filename = os.path.join(root_dir, f"dataset_analysis_{metadata['dataset']}.json")

    with open(filename, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Updated metadata with undersampled class info saved to {filename}")
