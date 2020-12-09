import numpy as np
import cv2
import tensorflow as tf
#(x_train, y_train), (x_test, y_test)=qtf.keras.datasets.mnist.load_data(path=r'C:\Users\KIIT\Desktop\smartknower')
model=tf.keras.models.load_model('mnist1.h5')
a=np.zeros([255,255],dtype='uint8')
wname='digit recognisation'
print("press p to predict")
print("press q to quit")
print("press c to clear screen")
cv2.namedWindow(wname)
b = False
def shape(event,x,y,flags,param):
    global b
    if event==cv2.EVENT_LBUTTONDOWN:
        b = True   
    elif(event==cv2.EVENT_MOUSEMOVE):
        if(b == True):
           cv2.circle(a,(x,y),5,(255,255,255),-8)
    elif event==cv2.EVENT_LBUTTONUP:
        b = False   
cv2.setMouseCallback(wname,shape)


while True:
    cv2.imshow(wname,a)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key== ord('c'):
       a[:,:]=0
    elif key== ord('p'):
        digit= a[50:250,50:250]
        digit=cv2.resize(digit,(28,28)).reshape(1,28,28)
        print("the predicted value is:")
        print(np.argmax(model.predict(digit)))
cv2.destroyAllWindows()    
