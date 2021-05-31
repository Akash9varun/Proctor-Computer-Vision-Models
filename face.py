import numpy as np
import cv2
import urllib.request

def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	# return the image
	return image

image = url_to_image("https://s3.amazonaws.com/nikeinc/assets/36722/Team-India-Kit-2015_16_9_hd_1600.jpg?1421176399")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#image = cv2.imread(img)
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
faces = face_cascade.detectMultiScale(grayImage)
  
#print (faces)
  
if len(faces) == 0:
    print ("No faces found")
  
else:
    #print (faces)
    print ("Number of faces detected: " + str(faces.shape[0]))
  
    #for (x,y,w,h) in faces:
    #    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
  
    #cv2.rectangle(image, ((0,image.shape[0] -25)),(270, image.shape[0]), (255,255,255), -1)
    #cv2.putText(image, "Number of faces detected: " + str(faces.shape[0]), (0,image.shape[0] -10), cv2.FONT_HERSHEY_TRIPLEX, 0.5,  (0,0,0), 1)
  
    #cv2.imshow('Image with faces',image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
