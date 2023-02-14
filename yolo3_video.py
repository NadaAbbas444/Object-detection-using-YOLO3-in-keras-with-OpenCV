import cv2
import numpy as np

######################################################################################################################
# the neural network configuration & the YOLO net weights file
# load the YOLO network
net = cv2.dnn.readNet('yolov3.cfg','yolov3.weights')
# loading all the class labels (objects)
with open('classes.txt', 'r') as f:
    classes =[]
    classes = f.read().splitlines()


img1 = cv2.VideoCapture(0)
i = 0
while img1.isOpened():
    fps = img1.get(cv2.CAP_PROP_FPS)
    print("Frames per second using img1.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    succ, frame = img1.read()
    height, width = frame.shape[:2]
    if succ == False:
        break
    # create 4D blob
    blob= cv2.dnn.blobFromImage(frame, 1 / 255, (415, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)  # sets the blob as the input of the network

    # get all the layer names
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
######################################################################################################################
    boxes = []
    confidences = []
    class_ids = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the object detections
        for detection in output:
            # extract the class id (label)& confidence (probability) of the current detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # discard weak predictions (less than 70%)
            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # (x, y)-coordinates of the top & left corner of the bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # update our list of bounding box coordinates, confidences,class IDs
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    # perform the non-maximum suppression given the scores defined before
    # idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    ######################################################################################################################
    # ensure at least one detection exists
    if len(indexes) > 0:
        # loop over the indexes we are keeping
        for i in indexes.flatten():
            # extract the bounding box coordinates
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 5)
            cv2.putText(frame, label + " " + confidence , (x, y + 20), font, 2, (0, 0, 255), 2)
    cv2.imshow('image', frame)
    i += 1
    cv2.waitKey(1)


