# data/dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FineGrainedDataset(Dataset):
    def __init__(self, root_dir, mode='train', known_classes=150, two_views=False, img_size=224):
        """
        Args:
            root_dir (str): Path to dataset root (should contain images and labels).
            mode (str): 'train', 'val', 'test_known', or 'test_unknown'.
            known_classes (int): Number of classes considered 'known' (others are treated as unknown).
            two_views (bool): If True, return two augmented views of each image (for contrastive pretraining).
            img_size (int): Desired image size (default is 224).
        """
        self.root_dir = root_dir
        self.mode = mode
        self.known_classes = known_classes
        self.two_views = two_views
        self.img_size = img_size  # 新增 img_size 参数

        # Prepare image file list and labels
        self.image_paths = []
        self.labels = []
        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        # Filter classes based on known/unknown for this mode
        if mode in ['train', 'val']:
            allowed_classes = {cls for cls, idx in self.class_to_idx.items() if idx < known_classes}
        elif mode == 'test_known':
            allowed_classes = {cls for cls, idx in self.class_to_idx.items() if idx < known_classes}
        elif mode == 'test_unknown':
            allowed_classes = {cls for cls, idx in self.class_to_idx.items() if idx >= known_classes}
        else:
            allowed_classes = set(self.class_to_idx.keys())

        for cls_name in classes:
            if cls_name not in allowed_classes:
                continue
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('jpg', 'jpeg', 'png')):
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls_name])

        # Determine transforms
        if mode == 'train' or mode == 'pretrain':
            # Data augmentation for training
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # using ImageNet means
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            # Validation / test: deterministic transform
            self.transform = transforms.Compose([
                transforms.Resize(int(self.img_size * 1.1)),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.two_views:
            # Return two augmented views of the same image for contrastive learning
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2, label
        else:
            img = self.transform(image)
            return img, label


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.two_views:
            # Return two augmented views of the same image for contrastive learning
            img1 = self.transform(image)
            img2 = self.transform(image)
            return img1, img2, label
        else:
            img = self.transform(image)
            return img, label

