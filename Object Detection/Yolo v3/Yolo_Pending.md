### ToDo

- How to decide the size of machine ? 
- The train operation fails when training on 14 GB machine, Coco128 dataset.
    - Separate operations of getitem from detections: create a separate flow and check if memory is increasing. 
    - Apply the 2-3 approaches found while searching
- Handle Grayscale Image
- Loss is not decreasing as of now. Look for solutions/alternatives and have learning graphs for the metrics
    - mAP
- Get away from 416, have a global variable
- Write function to download datasets and split them in test/val folders
- Write script to set up everything automatically
    - download yolo weights
    - set up conda
- Specify the path of images in a config file
- Train the network with a different set of classes 
- Check #TODO blocks