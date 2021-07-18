import cv2
import numpy as np
import math
from imutils.video import FileVideoStream
import os

weightsPath = os.path.join(os.path.dirname(
    __file__), 'utils/model/yolov4.weights')
cfgPath = os.path.join(os.path.dirname(__file__), 'utils/model/yolov4.cfg')
coco_namePath = os.path.join(os.path.dirname(__file__), 'utils/coco.names')


def load_model():
    # readNet is used to Read deep neural networks represented as weights and architectures
    # weight and cfg for DarkNet
    net = cv2.dnn.readNet(weightsPath, cfgPath)
    classes = []
    # Reading all classes
    with open(coco_namePath, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Storing all layers names in list
    layer_names = net.getLayerNames()
    # Identifying Output Layers
    output_layers = [layer_names[i[0]-1]
                     for i in net.getUnconnectedOutLayers()]

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return net, output_layers, classes

# Utility Functions


def findCentroid(xmin, ymin, xmax, ymax):
    xmid = (xmin+xmax)/2
    ymid = (ymin+ymax)/2
    centroid = (xmid, ymid)
    return centroid


def get_distance(x1, y1, x2, y2):
    dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
    return dist


def draw_detection_box(frame, x1, y1, x2, y2, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def main():
    VIOLET = (99, 20, 135)
    GREEN = (0, 168, 45)
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    WHITE = (255, 255, 255)
    GREY = (192, 192, 192)

    videoPath = os.path.join(os.path.dirname(__file__), 'TownCentre.mp4')
    video = FileVideoStream(videoPath).start()

    # video = cv2.VideoCapture(
    #     "C:/Users/vaibh/#VaibhaavRam/Intern/Social-Distancing-Detector-main/Social-Distancing-Detector-main/TownCentre.mp4")

    distance = 50  # According to social distance Norms, differs according to camera view

    net, output_layers, classes = load_model()
    while(video.more()):
        boxes = []
        confidences = []
        class_ids = []
        centroids = []
        box_colors = []
        safe_count, unsafe_count = 0, 0

        frame = video.read()

        frame_resize = cv2.resize(frame, (416, 416))

        height, width, channels = frame.shape

        blob = cv2.dnn.blobFromImage(
            frame_resize, 1/255, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                # Checking which of the objects in COCO the neural network detects more confidently
                class_id = np.argmax(scores)
                # Getting confidence value of that object
                confidence = scores[class_id]

                if(confidence > 0.3 and class_id == 0):  # Class_ID = 0 corresponds to Person
                    center_x = int(detection[0]*width)
                    # print(center_x)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    # Rectangular Coordinates
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non_MAX Suppression - which is used to combine overlapping bounding boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
        detectedBox = []

        for i in range(len(boxes)):
            if(i in indexes):
                x, y, w, h = boxes[i]
                xmin = x
                ymin = y
                xmax = x+w
                ymax = y+h
                centroid = findCentroid(xmin, ymin, xmax, ymax)
                # print(centroid)
                detectedBox.append([xmin, ymin, xmax, ymax, centroid])

                color = 0
                # Comparing each bounding box to every other
                for k in range(len(centroids)):
                    c = centroids[k]
                    p_x1 = int(c[0])
                    p_y1 = int(c[1])
                    p_x2 = int(centroid[0])
                    p_y2 = int(centroid[1])

                    # For drawing lines between two person standing close to each other
                    if(get_distance(p_x1, p_y1, p_x2, p_y2) <= distance):  # If two persons are too close
                        box_colors[k] = 1
                        color = 1
                        cv2.line(frame, (p_x1, p_y1),
                                 (p_x2, p_y2), YELLOW, 2, cv2.LINE_AA)
                        cv2.circle(frame, (p_x1, p_y1), 3,
                                   YELLOW, -1, cv2.LINE_AA)
                        cv2.circle(frame, (p_x2, p_y2), 3,
                                   YELLOW, -1, cv2.LINE_AA)
                        break

                centroids.append(centroid)
                box_colors.append(color)

        # To draw Bounding boxes
        for i in range(len(detectedBox)):
            x1 = detectedBox[i][0]
            y1 = detectedBox[i][1]
            x2 = detectedBox[i][2]
            y2 = detectedBox[i][3]

            if(box_colors[i] == 0):
                color = GREEN
                draw_detection_box(frame, x1, y1, x2, y2, color)
                label = "SAFE"
                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)

                y1label = max(y1, labelSize[1])
                cv2.rectangle(
                    frame, (x1, y1label - labelSize[1]), (x1+labelSize[0], y1+baseLine), WHITE, -1)
                cv2.putText(frame, label, (x1, y1),
                            cv2.FONT_HERSHEY_PLAIN, 1, GREEN, 1, cv2.LINE_AA)

                safe_count += 1

            else:
                color = RED
                draw_detection_box(frame, x1, y1, x2, y2, color)
                label = "UNSAFE"
                labelSize, baseLine = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 2)

                y1label = max(y1, labelSize[1])
                cv2.rectangle(
                    frame, (x1, y1label - labelSize[1]), (x1+labelSize[0], y1+baseLine), WHITE, -1)
                cv2.putText(frame, label, (x1, y1),
                            cv2.FONT_HERSHEY_PLAIN, 1, RED, 1, cv2.LINE_AA)

                unsafe_count += 1

        # To create Lengends
        cv2.rectangle(frame, (10, 10), (200, 60), GREY, -1)

        indication = "UNSAFE: "+str(unsafe_count)+" people"

        cv2.putText(frame, "--", (30, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.4, RED, 1, cv2.LINE_AA)
        cv2.putText(frame, indication, (60, 30),
                    cv2.FONT_HERSHEY_COMPLEX, 0.4, VIOLET, 1, cv2.LINE_AA)

        indication = "SAFE: "+str(safe_count)+" people"
        cv2.putText(frame, "--", (30, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.4, GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, indication, (60, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 0.4, VIOLET, 1, cv2.LINE_AA)

        frame = cv2.resize(frame, (1500, 700))

        cv2.imshow("HI", frame)
        cv2.waitKey(1)

        # if(cv2.waitKey(1) >= 0):
        #     break

    video.stop()
    # video.release()


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
