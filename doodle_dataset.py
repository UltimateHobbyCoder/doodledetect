import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class QuickDrawDataset(Dataset):
    def __init__(self, data_dir, samples_per_class=5000, transform=None):
        self.images = []
        self.labels = []
        self.class_names = []
        self.transform = transform

        files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        files.sort()  # for consistent label order

        for idx, filename in enumerate(files):
            class_name = filename.replace("full_numpy_bitmap_", "").replace(".npy", "")
            self.class_names.append(class_name)

            path = os.path.join(data_dir, filename)
            data = np.load(path)[:samples_per_class]
            data = data.reshape(-1, 28, 28).astype(np.uint8)

            self.images.extend(data)
            self.labels.extend([idx] * len(data))

        print(f"Loaded {len(self.images)} images from {len(self.class_names)} classes.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
