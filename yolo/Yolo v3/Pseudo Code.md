Yolo Implementation:

- Read and parse the config file: **parse_cfg**" function in **neural_net.py**
- Create a list (**module_list**) of all the layers found in parse_config file. For each layer, we'll have one item in the list. : **create_module_list** function in **neural_net.py**
    - *EmptyLayer* is used as a filler for shortcut, route and yolo layers. The pulling up of feature map (in case of shortcut layer) or detection calculations (in case of yolo layer) will be handled in the forward function of the network.
- Create a network from it the parsed cfg file
    - init function: **__init()__** in **neural_net.py**
        - initialize layer_dic_list, module_list and net_info
    - forward function: **forward** in **neural_net.py**
        - create a "global_detections" tensor
        - keep a track of feature maps as it'll be required to source input for shortcut and route layers: feature_map_list
        - conv block
        - shortcut block
        - route block
        - upsample block
        - yolo block:
            - get the anchors for this particular yolo layer. These anchors would be used for calculating the width and height of each bounding box.: **get_anchors** function in **neural_net.py**
            - depending upon the yolo layer, output would be of different size. perform necessary calculations, extract relevant values from the feature map: **transform_yolo_output** in **neural_net.py**
                - Repeat the anchor tensor grid_size*grid_size times along first dimension. We will need this tensor later while performing yolo calculations.
                - Divide the anchor tensor by stride --> if input has shrunk by a factor of stride, so should the anchors.
                - output is in the form of batch_size x 255 x SxS grid. We'll have to loop through each item in the batch. [S = *grid_size* above]
                    - The number 255 corresponds to total values in a cell: 85 x 3. Transpose the input to SxSx255 
                    - Reshape the grid in order to have only 2 dimensions. Dimensions would be S*S x 255
                    - each row in this tensor contains data for 3 BBs. Again reshape the above tensor: one row for one bounding box. The dimensions now would be (S*S\*3) x 85
                    - apply sigmoid on first, second, and fifth element of each row
                    - create a grid with the values of cx and cys. Remember that 3 bounding boxes belong to one cell. First 3 elements belong to (0,0) coordinate, next 3 belong to (1,0) coordinate, and so on. Hence, the grid would have to be created accordingly. : **get_mesh_grid()** function in **neural_net.py**
                    - do the bounding box calculations (bx = cx + sigmoid(tx), and other such calculations)
                    - remove the rows that have fifth element below the detection threshold: these rows don't have any detection box
                    - convert bx,by,bw,bh to bx1, bx2, by1, by2 ==> from center coordinates and width, height to box coordinates
                    - for each row, reduce the number of elements to 7: bx1, by1, bx2, by2, the class index this bb corresponds to, the confidence in that class. the last two values will retrieved by using max function (one bb can have only one object. the class having max probab is the class detected)

                    // at this time, we have only those bbs which have some probability of detecting an element. The bbs have 7 coordinates as explained above. One problem to solve for: same element can be detected by multiple boxes (not cells, boxes)
                    - initialize a flag if detection tensor exists
                    - create a list of total unique classes detected
                    - loop over this classes list:
                        - get all the rows corresponding to each class
                        - sort in desc order (wrt to confidence in that class)
                        - calculate iou of each row with all other rows ==> generates an iou_tensor, the size of tensor being num_detections_for_each_class * num_detections_for_each_class
                        - create two lists: one for rejected indices and another for detected indices.
                        - for each row in iou_tensor:
                            - [ToDo] have a helper function to generate an image with all bbs drawn at this stage
                            - if row index in rejected index list, continue
                            - get the indices of all the columns in which iou is greater than iou_threshold
                            - add items of above list to rejection list ==> as the elements are sorted in descending order, if iou > threshold, the indexes correspond to same object
                            - add the current index to detection list ==> index of detection having highest confidence. 
                        - multiply the detection tensor with stride in order to scale to the final image
                        - create (or add to) a tensor that contains all the detections.
                - add to the detections tensor to global_detection dict. A key in this dict would be the index of the batch, value would be a tensor with each row corresponding to a detection, 7 columns corresponding to attributes defined earlier.

- load the weights (follow Akshay Kathooriya's blog)
- read an image and pass through it to get the results: **image_to_tensor** in **neural_net.py**
- write a function to draw bounding boxes and name of the class: **draw_rectangle()** in **neural_net.py**

- Write a shell script to download weights file
- Automate conda env creation