from typing import Optional
from fastapi import FastAPI
import numpy as np
import cv2
import urllib.request
from pydantic import BaseModel

app = FastAPI()

class DataModel(BaseModel):
    url:str


print("LOADING YOLO")
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
#save all the names in file o the list classes
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
#get layers of the network
layer_names = net.getLayerNames()
#Determine the output layer names from the YOLO model 
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
print("YOLO LOADED")

def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image

@app.get('/')
def serverstatus():
    return {"status": "Running"}

@app.post("/faces/")
def read_root(dm:DataModel):
    baseurl = dm.url
    image = url_to_image(baseurl)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayImage)
    return {"NoofFaces":faces.shape[0] }



@app.post("/mobile/")
def read_mobile(dm:DataModel):
    baseurl = dm.url
    img = url_to_image(baseurl)
    height, width, channels = img.shape

    # USing blob function of opencv to preprocess image
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
    swapRB=True, crop=False)
        #Detecting objects
    net.setInput(blob)
    outs = net.forward(output_layers)

        # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                    # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
        
        #We use NMS function in opencv to perform Non-maximum Suppression
        #we give it score threshold and nms threshold as arguments.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    response=[]
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            #cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            #cv2.putText(img, label, (x, y -5),cv2.FONT_HERSHEY_SIMPLEX,
            #1/2, color, 2)
            #print(label)
            response.append(label)
    return {"result": response}
          
        
    
