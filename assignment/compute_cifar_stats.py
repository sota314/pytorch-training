import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def compute_cifar10_mean_std(batch_size: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(root="./", train=True, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    n_batches = 0
    channel_sum = torch.zeros(3)
    channel_sum_sq = torch.zeros(3)
    total_images = 0

    for images, _ in loader:
        b, c, h, w = images.shape
        total_images += b
        n_batches += 1

        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])

    #平均と標準偏差の計算
    num_pixels_per_channel = total_images * h * w
    mean = channel_sum / num_pixels_per_channel
    std = (channel_sum_sq / num_pixels_per_channel - mean ** 2).sqrt()

    return mean, std


def main() -> None:
    mean, std = compute_cifar10_mean_std()
    print("CIFAR-10 train mean:", tuple(mean.tolist()))
    print("CIFAR-10 train std :", tuple(std.tolist()))


if __name__ == "__main__":
    main()
