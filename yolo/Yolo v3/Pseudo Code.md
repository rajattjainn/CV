Yolo Implementation:

- Read and parse the config file: **parse_cfg**" function in **neural_net.py**
- Create a list (**module_list**) of all the layers found in parse_config file. For each layer, we'll have one item in the list. : **create_module_list** function in **neural_net.py**
    - *EmptyLayer* is used as a filler for shortcut and yolo layer. The pulling up of feature map (in case of shortcut layer) or detection calculations (in case of yolo layer) will be handled in the forward function of the network.
- create a "global_detections" tensor
- Create a network from it the parsed cfg file
    - net layer
    - conv layer
    - shortcut layer
    - route layer
    - upsample layer
    - yolo layer
- Implement yolo layer:
    - output is in the form of a sxs grid. flatten this into a column tensor
    - each row in this tensor contains data for 3 BBs
    - again flatten the above column tensor: one row for one bounding box
    - apply sigmoid on first, second, and fifth element of each row
    - create a grid with the values of cx and cys
    - do the appropriate repeats 
    - do the bounding box calculations (bx = cx + sigmoid(tx), and other such calculations)
        - make sure to perform assert checks
    - zero the elements that have fifth element below the detection threshold
    - remove the elements that have zero value for 5th element
    - for each row, reduce the number of elements to 7: bx, by, bw, bh, the class index this bb corresponds to, the confidence in that class. the last two values will retrieved by using max function (one bb can have only one object. the class having max probab is the class detected)

    // at this time, we have only those bbs which have some probability of detecting an element. The bbs have 7 coordinates as explained above. One problem to solve for: same element can be detected by multiple boxes (not cells, boxes)
    - initialize a "detections" tensor
    - multiply the detection tensor with stride in order to scale to the final image
    - create a list of total unique classes detected
    - loop over this classes list:
        - get all the rows corresponding to each class
        - sort in desc order (wrt to confidence in that class)
        - for each item of this list:
            - have a helper function to generate an image with all bbs drawn at this stage
            - compare each item with the remaining/following items, calculate iou 
                - If iou > thres, discard that particular (with lesser probability) item from the list
                - if iou< thres, let that item be in the list. this means the object is a separate object
                - be aware of index out of bounds error
            - at the endof this for loop, add the items of this list (the list corresponding to a specific class) to the detection tensor
        - after iterating over the list, we'll have all the items detected in that layer for all the classes
    - add to the detections tensor to global_detection tensor
- the global_detection tensor will have each row corresponding to one object. It'll have 7 columns: bx, by, bw, bh, the class index this bb corresponds to, the confidence in that class

- load the weights (follow Akshay Kathooriya's blog)
- read an image and pass through it to get the results
- write a function to draw bounding boxes and name of the class

- Write a shell script to download weights file
- Automate conda env creation