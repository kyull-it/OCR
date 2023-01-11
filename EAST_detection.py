import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import argparse

parser = argparse.ArgumentParser(description = "Create an instant that extract the image path you will input for EAST detection.")
parser.add_argument('-i', '--image', required=True, help = "Please enter the image path to be used for EAST detection.")
args = parser.parse_args()

imagepath = args.image
imagename = imagepath.split('/')[-1].split('.')[0]
print(imagename)

image = cv2.imread(imagepath)

resize_num = 320


def resize(image):
    
    re_image = cv2.resize(image, (resize_num, resize_num), cv2.INTER_CUBIC)
    print(image.shape, " --> ", re_image.shape)
    return re_image


from imutils.object_detection import non_max_suppression

def getResults(image):
    
    width, height = image.shape[:2]
    min_confidence = 0.5
    
    m1 = np.mean(image[0])
    m2 = np.mean(image[1])
    m3 = np.mean(image[2])

    blob = cv2.dnn.blobFromImage(image, 1.0, (width, height), (m1, m2, m3), swapRB=True)
    
    net = cv2.dnn.readNet('model/frozen_east_text_detection.pb')
    print("Loading the east model : ", net)
    
    net.setInput(blob)
    
    layers = ["feature_fusion/Conv_7/Sigmoid", 
          "feature_fusion/concat_3"]

    (scores, geometry) = net.forward(layers)

    print("Success to load the model.")
    print(".")
    print("[Scores]")
    print(scores.shape)
    print("[Geometry]")
    print(geometry.shape)

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    
    for y in range(0, numRows):
        
        scoreData = scores[0,0,y]
        gTop = geometry[0,0,y]
        gRight = geometry[0,1,y]
        gBottom = geometry[0,2,y]
        gLeft = geometry[0,3,y]
        gAngle = geometry[0,4,y]
        
        for x in range(0, numCols):
            
            if scoreData[x] < min_confidence:
                continue
            
            angle = gAngle[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            
            h = gTop[x] + gBottom[x]
            w = gRight[x] + gLeft[x]
            
            # model result must be 4x smaller than original.
            offset = 4.0
            
            endX = int(offset * x + (cos * gRight[x]) + (sin * gBottom[x]))
            endY = int(offset * y + (sin * gRight[x]) + (cos * gBottom[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            
            rects.append((startX, startY, endX, endY))
            confidences.append(scoreData[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)   
    
    return boxes


def drawBoxes(boxes, image):
    
    height, width = image.shape[:2]
    rW = round(width/resize_num, 5)
    rH = round(height/resize_num, 5)
    
    padding = 10
    
    for (startX, startY, endX, endY) in boxes:
        
        # startX = math.floor(startX * rW)
        # startY = math.floor(startY * rH)
        # endX = math.ceil(endX * rW)
        # endY = math.ceil(endY * rH)
        
        # startX = math.floor(startX * rW) - int(width*padding)
        # startY = math.floor(startY * rH) - int(height*padding)
        # endX = math.ceil(endX * rW) + int(width*padding)
        # endY = math.ceil(endY * rH) + int(height*padding)
        
        startX = math.floor(startX * rW) - padding
        startY = math.floor(startY * rH) - padding
        endX = math.ceil(endX * rW) + padding
        endY = math.ceil(endY * rH) + padding
        
        cv2.rectangle(image, (startX, startY), (endX, endY), (0,255,0), 3)
        
    return image


inputimage = resize(image)
boxes = getResults(inputimage)
result = drawBoxes(boxes, image)

cv2.imwrite('./results/result_'+imagename+'(320).jpg', result)

cv2.imshow('Image', result)
cv2.waitKey(0) 
cv2.destroyAllWindows()
