from torchvision import transforms, datasets

def cifar_datasets():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_data = datasets.CIFAR10(
        root='./',
        train=True,
        transform=train_transform,
        download=True
        )

    test_data = datasets.CIFAR10(
        root='./',
        train=False,
        transform=test_transform,
        download=True
    )
    return train_data, test_data

if __name__=='__main__':
    train_data, test_data = cifar_datasets()

    image, label = train_data[0]
    print(f"image size: {image.size()}")
    print(f'label: {label}')