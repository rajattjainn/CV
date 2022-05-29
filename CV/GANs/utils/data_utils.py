import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

def get_datloader(data_dir, image_size, batch_size, shuffle, num_workers):
    """
    Create a dataset and dataloader from data_dir, returns dataloader.

    Keyword arguments:
    data_dir: the data directory for which dataloader is to be created.
    image_size: the size each image should have. This is used for rescaling.
    batch_size: the size of the batch for dataloader.
    shuffle: whether data should be shuffled or not in the dataloader.
    num_workers: number of worker threads to use.

    Returns:
    dataloader: the dataloader created from data_dir
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
