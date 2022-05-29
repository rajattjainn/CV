import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

def get_datloader(data_dir, image_size, batch_size, shuffle, num_workers):
    """
    """
    dataset = dset.ImageFolder(root=data_dir, 
                    transform = transforms.Compose(
                        [
                            transforms.Resize(image_size),
                            transforms.CenterCrop(image_size),
                            transforms.ToTensor(),
                            transforms.normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                        ]
                    ))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, 
                                        shuffle = shuffle, num_workers = num_workers)
    
    return dataloader
