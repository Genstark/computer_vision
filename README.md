# ObjectDetector
An Object Detector made using Python, OpenCV, Mobilenet SSD v3 and Tensor Flow

This Object Detector program involves OpenCV for most Image related operations and the Trained Model used is taken from TensorFlow. I Have used the MobileNet SSD for the Detection and this detector detects objects with the minimum Confidence level of 50% and hence lite weight. 

Download the Config and Weights files from here
- https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API

The COCO Dataset(Common Objects in Context)
- https://cocodataset.org/#home
- https://github.com/pjreddie/darknet/blob/master/data/coco.names

FILES USED
- labels.txt - List of 80 Labels that the Detected Objects are Classified Under.
- frozen_inference_graph.pb - The protbuf file containing the graph definition as well as the weights of the model.
- ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt - The Config File.
