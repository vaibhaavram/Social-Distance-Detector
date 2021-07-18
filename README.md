# Social-Distance-Detector

This project uses YOLO object detection which is a pretrained model to detect persons in video streams using OpenCV. The detected persons are bounded with bounding boxes. Centroid is calculated for each bounding box and if the distance between two bounding boxes is lesser than certain value, then they are connected with a Line and they
are labelled "Unsafe". If they are far away from each other, they are labelled "Safe". Also there is a provision to keep track of the count of Safe and Unsafe people and it is displayed in the legend.
