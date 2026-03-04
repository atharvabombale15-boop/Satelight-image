import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class LEVIRDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        image_size=256,
        hard_negative_ratio=0.0,
        negative_threshold=0.005,
    ):
        self.t1_dir = os.path.join(root_dir, split, "t1")
        self.t2_dir = os.path.join(root_dir, split, "t2")
        self.mask_dir = os.path.join(root_dir, split, "mask")

        self.image_names = sorted(os.listdir(self.t1_dir))
        self.indices = list(range(len(self.image_names)))

        self.transform_img = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.transform_mask = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        if hard_negative_ratio > 0:
            self._apply_hard_negative_sampling(hard_negative_ratio, negative_threshold)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_name = self.image_names[self.indices[idx]]

        t1_path = os.path.join(self.t1_dir, img_name)
        t2_path = os.path.join(self.t2_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        t1 = Image.open(t1_path).convert("RGB")
        t2 = Image.open(t2_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        t1 = self.transform_img(t1)
        t2 = self.transform_img(t2)
        mask = self.transform_mask(mask)

        mask = (mask > 0).float()  # ensure binary

        return t1, t2, mask

    def _apply_hard_negative_sampling(self, hard_negative_ratio, negative_threshold):
        negatives = []
        positives = []
        for i, name in enumerate(self.image_names):
            mask_path = os.path.join(self.mask_dir, name)
            mask = Image.open(mask_path).convert("L")
            mask_tensor = self.transform_mask(mask)
            change_ratio = float((mask_tensor > 0).float().mean().item())
            if change_ratio <= negative_threshold:
                negatives.append(i)
            else:
                positives.append(i)

        if not negatives:
            self.indices = list(range(len(self.image_names)))
            return

        repeat = max(1, int(1 + hard_negative_ratio))
        self.indices = positives + negatives * repeat
