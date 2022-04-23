This is a PyTorch implementation of Yolo v3 algorithm written by Joseph Redmon and Ali Farhadi.
Yolo is a family of Object Detection Algorithms and Yolo v3 is one of the most successful algorithms out there.

### Understanding the Project
1. A guide to understand can be found here: [Yolo Theory](Yolo_Theory.md).
2. The algorithm and detection code can found in [detect.py](detect.py), [neural_net.py](neural_net.py), and [utils.py](utils.py). I was planning to write a code walkthrough guide but later decided that my time would be better spent if I document the code properly.
3. A guide to understand teh training mechanism can be found here: [Yolo Train](Yolo_Train.md).

### Running the Project
1. The [environment.yml](environment.yml) file has all the dependencies for this project. It is adviced to create a conda/miniconda environment, install all the dependencies.
2. You'll also have to download the weights. You can do that by running the following command:
> wget https://pjreddie.com/media/files/yolov3.weights
3. In order to detect objects, you have to run detect.py file. Remember to update the value of *image_dir_path* variable before running the file.
4. In order to train the network, you have to run *train.py* file. Just like while detecting, you'll have to update the locations of train/eval images and labels.

#### Pending Tasks
A bunch of tasks are pending before I close this project. All those tasks could be found here: [Pending Tasks](Yolo_Pending.md)