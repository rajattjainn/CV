import torchvision.transforms as transforms

def get_image_transform():
    image_transform = transforms.Compose([
            transforms.Resize([416, 416]),
            transforms.ToTensor()])

    return image_transform