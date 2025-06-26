"""
This module contains the datasets used in the AutoML exam.
If you want to edit this file be aware that we will later 
  push the test set to this file which might cause problems.
"""

from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import PIL.Image
import pandas as pd
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, check_integrity
import tempfile
import shutil

# URL to the zip file containing all phase 1 datasets
BASE_URL = "https://ml.informatik.uni-freiburg.de/research-artifacts/automl-exam-25-vision/"
ZIP_FILE = "vision-phase1.zip"


class BaseVisionDataset(VisionDataset):
    """A base class for all vision datasets.

    Args:
        root: str or Path
            Root directory of the dataset (should contain the extracted datasets).
        split: string (optional)
            The dataset split, supports `train` (default), `val`, or `test`.
        transform: callable (optional)
            A function/transform that takes in a PIL image and returns a transformed version. 
            E.g, `transforms.RandomCrop`.
        target_transform: callable (optional)
            A function/transform that takes in the target and transforms it.
        download: bool (optional)
            If true, downloads the dataset zip and extracts it into the root directory.
            If dataset is already downloaded, it is not downloaded again.
    """
    _download_url_prefix = BASE_URL
    _download_file = ZIP_FILE
    _dataset_name: str
    width: int
    height: int
    channels: int
    num_classes: int

    def __init__(
        self,
        root: Union[str, Path] = "data",
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        assert split in ["train", "test"], f"Split {split} not supported"
        self._split = split
        self._base_folder = Path(self.root) / self._dataset_name

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. You can use download=True to download it "
                f"or download it manually from {self._download_url_prefix}{self._download_file}"
            )

        data = pd.read_csv(self._base_folder / f"{self._split}.csv")
        self._labels = data['label'].tolist()
        self._image_files = data['image_file_name'].tolist()

    def _check_integrity(self) -> bool:
        train_images_folder = self._base_folder / "images_train"
        test_images_folder = self._base_folder / "images_test"
        # Check if image folders exist
        if not (train_images_folder.exists() and train_images_folder.is_dir()) or \
           not (test_images_folder.exists() and test_images_folder.is_dir()):
            return False

        # Check if csv files exist
        if not (self._base_folder / "train.csv").exists() or not (self._base_folder / "test.csv").exists():
            return False

        return True

    def download(self) -> None:
        """Download and extract the zip file containing all datasets into root directory."""
        if self._check_integrity():
            print("Dataset already downloaded and verified.")
            return

        # Extract to temporary location
     
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Downloading and extracting {self._download_file}...")
            
            download_and_extract_archive(
                url=f"{self._download_url_prefix}{self._download_file}",
                download_root=temp_dir,
                extract_root=temp_dir,
                filename=self._download_file,
                remove_finished=True
            )
            
            # Find phase folder and move contents to data/
            phase_folder = next(Path(temp_dir).glob("phase*"))
            data_path = Path(self.root)
            data_path.mkdir(exist_ok=True)
            
            for item in phase_folder.iterdir():
                destination = data_path / item.name
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.move(str(item), str(destination))
        
        print("Download completed.")
    def extra_repr(self) -> str:
        """String representation of the dataset."""
        return f"split={self._split}"

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image_path = self._base_folder / f"images_{self._split}" / image_file
        image = PIL.Image.open(image_path)
        if self.channels == 1:
            image = image.convert("L")
        elif self.channels == 3:
            image = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported number of channels: {self.channels}")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self) -> int:
        return len(self._image_files)


class EmotionsDataset(BaseVisionDataset):
    """ Emotions Dataset.

    This dataset contains images of faces displaying in to one of seven emotions
    (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
    """
    _dataset_name = "emotions"
    width = 48
    height = 48
    channels = 1
    num_classes = 7


class FlowersDataset(BaseVisionDataset):
    """Flower Dataset.

    This dataset contains images of 102 types of flowers. The task is to classify the flower type.
    """
    _dataset_name = "flowers"
    width = 512
    height = 512
    channels = 3
    num_classes = 102


class FashionDataset(BaseVisionDataset):
    """Fashion Dataset.

    This dataset contains images of fashion items. The task is to classify what kind of fashion item it is.
    """
    _dataset_name = "fashion"
    width = 28
    height = 28
    channels = 1
    num_classes = 10
