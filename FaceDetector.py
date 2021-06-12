import numpy as np
import argparse
import cv2

# define the argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path tp Caffe model file")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="min probability to filter weak detection")
args = vars(ap.parse_args())

print("[INFORMATION] Loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# load the input image and construct an input blob for the image
# by resizing to a fixed 900x900 pixels and then normalizing it
img = cv2.imread("D.png")
h = img.shape[0]
w = img.shape[1]
blob = cv2.dnn.blobFromImage(cv2.resize(img, (900, 900)), 1.0,
                            (900, 900), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence
    confidence = detections[0, 0, i, 2]
    
    # filter out weak detections by ensuring the 'confidence' is greater than minimum confidence
    if confidence > args["confidence"]:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box with the probability
        text = "{:.2f}%".format(confidence*100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(img, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)

cv2.imshow("Face_Detection", img)
cv2.waitKey(0)

# launch: python FaceDetector.py --prototxt deploy.prototxt.txt --model res10_300X300_ssd_iter_140000.caffemodel