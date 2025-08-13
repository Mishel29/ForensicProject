import os
import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob

class FingerprintDataset(Dataset):
    def __init__(self, root_dir):
        self.latent_paths = sorted(glob(os.path.join(root_dir, 'latent', '*.png')))
        self.skeleton_paths = sorted(glob(os.path.join(root_dir, 'skeleton', '*.png')))
        self.orientation_paths = sorted(glob(os.path.join(root_dir, 'orientation', '*.npy')))
        assert len(self.latent_paths) == len(self.skeleton_paths) == len(self.orientation_paths), "Mismatch in data files"

    def __len__(self):
        return len(self.latent_paths)

    def __getitem__(self, idx):
        latent = self._load_image(self.latent_paths[idx])
        skeleton = self._load_image(self.skeleton_paths[idx])
        orientation = self._load_npy(self.orientation_paths[idx])
        return latent, skeleton, orientation

    def _load_image(self, path):
        import cv2
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img)

    def _load_npy(self, path):
        import cv2
        arr = np.load(path)
        arr = cv2.resize(arr, (256, 256), interpolation=cv2.INTER_NEAREST)
        arr = arr.astype(np.float32) / np.pi  # normalize orientation to [0,1]
        return torch.from_numpy(arr)
