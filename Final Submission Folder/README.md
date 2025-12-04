This Folder Contains the Source Code for the HAzard Distance Detection Computer Vision Final Project

The Pinhole Directory:
This directory contains 

Pinhole_Object_Detection_Pipeline Directory:
This directory contains a handful of files and sub directories.

Files:
    - object_detect_orthophoto.py: The main program that loads the input image, runs object detection, creates an orthophoto, and utilizes world coordinates to calculate distance of objects. (Reuses most of the code from pinhole.py)

    - open_vocab_detect_v2.py: This is the object detetion model that is ran as part of the entie pipeline. This model utilizes a list of labels to classify object within the image.

    - label.txt: List of all labels being check for within the image. Can be expanded to get more hazards, but this was a short list for basic testing.

    - error.py: A short script that produced the barcharts used in the report for visualsing accuary of the pipeline

    - distance_comparison.png: Sample barchart produced by error.py


Subdirectories:
    - INPUT: Contains all of the images used during testing and development.
    - OUTPUT: Contains all of the outputs produced by image detection as well as The orthophoto

To run the entire pinhole model with object detection pipeline the command "python object_detect_orthophoto.py" is all that is needed. 

Note *At the moment specifying an input image requires manually inputing the path to the file in the main function of the program. This is due to each input image having different camera intrinsics and extrinsicts that need to be manually set depending on the camera that the image was taken on as well as the enviroment the image was taken in*



