As the custom is, the authors of Yolo v3 have open-sourced the weights file of the network. We used the same file while detecting various objects. However, if our dataset does not contain the same class of objects, we can train the network so that it is able to detect the objects of our interests. The 


We use "transfer learning" in order to enable the neural network to detect objects of our interest. I have broken down the problem of training the neural network into various steps as follows:

1. Data preparation
2. 

### Understanding the Data
The training data is split into two parts - image data and label data. Each type of data belonging to a separate folder (i.e. images and labels). For each image file, we have a corresponding label file.
An image can have multiple objects. Let's say an image has 3 objects - dog, cat, and mouse. The corresponding label file in this case would have 3 rows, each row corresponding to one object. The structure of a single row in a label file is as follows:

> object_class, x_center, y_center, width, height

- **object_class** is an integer corresponding to the class ranging from 0 to num_classes-1.
- **x_center** and  **y_center** are the relative center coordinates of the object. By relative, we mean that the center coordinates have been divided by the width and height of the image. This means that these coordinates will always fall between the range of 0 and 1.
- **width** and **height** are the relative (as in the case of x_center and y_center) width and height of the object. 


Let's take example of Coco128 dataset. 