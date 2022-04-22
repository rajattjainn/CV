As the custom is, the authors of Yolo v3 have open-sourced the weights file of the network. We used the same file while detecting various objects. However, if our dataset does not contain the same class of objects, we can train the network so that it is able to detect the objects of our interests. The 


We use "transfer learning" in order to enable the neural network to detect objects of our interest. I have broken down the problem of training the neural network into various steps as follows:

1. Data preparation
2. Understanding Loss for Yolo v3
3. Implementing Loss function
4. Results

### Understanding the Data
The training data is split into two parts - image data and label data. Each type of data belonging to a separate folder (i.e. images and labels). For each image file, we have a corresponding label file.
An image can have multiple objects. Let's say an image has 3 objects - dog, cat, and mouse. The corresponding label file in this case would have 3 rows, each row corresponding to one object. The structure of a single row in a label file is as follows:

> object_class, x_center, y_center, width, height

- **object_class** is an integer corresponding to the class ranging from 0 to num_classes-1.
- **x_center** and  **y_center** are the relative center coordinates of the object. By relative, we mean that the center coordinates have been divided by the width and height of the image. This means that these coordinates will always fall between the range of 0 and 1.
- **width** and **height** are the relative (as in the case of x_center and y_center) width and height of the object. 


### Understanding Loss for Yolo v3
To train the neural network, we need a loss function. To (re)train Yolo v3, we'll use the same loss function as used by the authors. The authors have divided the loss into three parts:

1. **Coordinate Loss**: the loss between the ground truth coordinates (i.e. bounding box) and the predicted coordinates.
2. **Class Loss**: The loss in the class predicted of the object and the class of the Ground Truth object.
3. **Confidence Loss**: The loss in the confidence/objectness between the predicted value and the actual value.

Of the above three, the first two are calculate only for the ground truth objects and the corresponding predicted objects. The third one, i.e. Confidence Loss is calculate for all the predictions.

After calculating the individual losses, we sum them up; the summation constitutes the total loss for each prediction.

**Further explanation for Confidence Loss:** I had some problem while implementing the confidence loss. My confusion was that the total number of predictions would be much greater than the actual ground truth boxes. So how will this loss be calculated ? <br>
The predicted tensor has a confidence value for each prediction, the 5th column for each row. We also have a IoU tensor which captures the IoU value between the predicted and the ground truth boxes. If we look closely, the IoU tensor serves as a proxy for the ground truth boxes - it has high IoU value for those predicted boxes which are actual ground truth boxes. <br> 
What we do is that we filter out this column vector with a threshold value - all the values above the threshold are true predictions and all those that are below serve as false predictions. Now we calculate the loss function between this and the prediction tensor (without the iou calculated over it). 

### Implementing Loss function
The functions to calculate loss can be found in [utils.py](utils.py) class:  calculate_loss and individual_loss. Both the functions are self-explanatory and have in-line comments added.

### Results
[Update April 22]: Loss isn't decreasing after the first epoch. Need to dig deeper