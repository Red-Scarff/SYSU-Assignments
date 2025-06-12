"""
Data augmentation strategies for semi-supervised learning
Includes RandAugment for strong augmentation in FixMatch
"""

import random
import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance, ImageDraw
import torchvision.transforms as transforms


class RandAugmentMC:
    """
    RandAugment implementation for FixMatch
    Applies n random augmentations with magnitude m
    """

    def __init__(self, n=2, m=10):
        self.n = n  # Number of augmentations to apply
        self.m = m  # Magnitude of augmentations
        self.augment_list = self._get_augment_list()

    def _get_augment_list(self):
        """Get list of available augmentations with their ranges"""
        return [
            (self._auto_contrast, 0, 1),
            (self._brightness, 0.05, 0.95),
            (self._color, 0.05, 0.95),
            (self._contrast, 0.05, 0.95),
            (self._equalize, 0, 1),
            (self._identity, 0, 1),
            (self._posterize, 4, 8),
            (self._rotate, -30, 30),
            (self._sharpness, 0.05, 0.95),
            (self._shear_x, -0.3, 0.3),
            (self._shear_y, -0.3, 0.3),
            (self._solarize, 0, 256),
            (self._translate_x, -0.3, 0.3),
            (self._translate_y, -0.3, 0.3),
        ]

    def __call__(self, img):
        """Apply random augmentations to image"""
        ops = random.choices(self.augment_list, k=self.n)
        for op, min_val, max_val in ops:
            val = min_val + float(max_val - min_val) * random.random()
            img = op(img, val)

        # Apply cutout with random magnitude
        cutout_val = random.random() * 0.5
        img = self._cutout(img, cutout_val)
        return img

    def _auto_contrast(self, img, _):
        return ImageOps.autocontrast(img)

    def _brightness(self, img, v):
        return ImageEnhance.Brightness(img).enhance(v)

    def _color(self, img, v):
        return ImageEnhance.Color(img).enhance(v)

    def _contrast(self, img, v):
        return ImageEnhance.Contrast(img).enhance(v)

    def _equalize(self, img, _):
        return ImageOps.equalize(img)

    def _identity(self, img, _):
        return img

    def _posterize(self, img, v):
        v = int(v)
        v = max(1, v)
        return ImageOps.posterize(img, v)

    def _rotate(self, img, v):
        return img.rotate(v)

    def _sharpness(self, img, v):
        return ImageEnhance.Sharpness(img).enhance(v)

    def _shear_x(self, img, v):
        return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))

    def _shear_y(self, img, v):
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))

    def _solarize(self, img, v):
        return ImageOps.solarize(img, int(v))

    def _translate_x(self, img, v):
        v = v * img.size[0]
        return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))

    def _translate_y(self, img, v):
        v = v * img.size[1]
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))

    def _cutout(self, img, v):
        """Apply cutout augmentation"""
        if v <= 0.0:
            return img

        v = v * img.size[0]
        return self._cutout_abs(img, v)

    def _cutout_abs(self, img, v):
        """Apply cutout with absolute size"""
        if v < 0:
            return img

        w, h = img.size
        x0 = np.random.uniform(w)
        y0 = np.random.uniform(h)

        x0 = int(max(0, x0 - v / 2.0))
        y0 = int(max(0, y0 - v / 2.0))
        x1 = min(w, x0 + v)
        y1 = min(h, y0 + v)

        xy = (x0, y0, x1, y1)
        color = (125, 123, 114)  # CIFAR-10 mean color
        img = img.copy()
        ImageDraw.Draw(img).rectangle(xy, color)
        return img


class TransformTwice:
    """Apply the same transform twice (for MixMatch)"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class TransformFixMatch:
    """
    Dual augmentation for FixMatch: weak and strong
    """

    def __init__(self, mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)):
        self.weak = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode="reflect"),
            ]
        )

        self.strong = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode="reflect"),
                RandAugmentMC(n=2, m=10),
            ]
        )

        self.normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        # x should be a PIL Image (like reference implementation)
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class RandomPadandCrop:
    """Random pad and crop like in reference implementation"""

    def __init__(self, output_size):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, x):
        # Pad with 4 pixels
        x = np.pad(x, [(0, 0), (4, 4), (4, 4)], mode="reflect")

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top : top + new_h, left : left + new_w]
        return x


class RandomFlip:
    """Random horizontal flip like in reference implementation"""

    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]
        return x.copy()


class ToTensorFromNumpy:
    """Convert numpy array to tensor"""

    def __call__(self, x):
        return torch.from_numpy(x.copy())


def get_cifar10_transforms():
    """Get CIFAR-10 transforms compatible with reference implementation"""

    # Training transform (for data already normalized and transposed)
    train_transform = transforms.Compose(
        [
            RandomPadandCrop(32),
            RandomFlip(),
            ToTensorFromNumpy(),
        ]
    )

    # Test transform (for data already normalized and transposed)
    test_transform = transforms.Compose(
        [
            ToTensorFromNumpy(),
        ]
    )

    return train_transform, test_transform


def get_cifar10_transforms_standard():
    """Get standard CIFAR-10 transforms for PIL images"""

    # CIFAR-10 normalization constants
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)

    # Training transform (weak augmentation)
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # Test transform (no augmentation)
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    return train_transform, test_transform


if __name__ == "__main__":
    # Test augmentations
    from PIL import Image
    import numpy as np

    # Create a dummy image
    img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    # Test RandAugment
    rand_aug = RandAugmentMC(n=2, m=10)
    augmented = rand_aug(img)
    print(f"Original image size: {img.size}")
    print(f"Augmented image size: {augmented.size}")

    # Test FixMatch transforms
    fixmatch_transform = TransformFixMatch()
    weak, strong = fixmatch_transform(img)
    print(f"Weak augmentation shape: {weak.shape}")
    print(f"Strong augmentation shape: {strong.shape}")
