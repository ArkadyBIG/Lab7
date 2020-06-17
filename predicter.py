import cv2
import matplotlib.pyplot as plt
import numpy

from tensorflow.keras import models
model = models.load_model('input-300-6.h5')

cam = cv2.VideoCapture(0)
cv2.namedWindow("test")
img_counter = 0

while True:
    ret, frame = cam.read()
    # frame (480, 640, 3)
    x0 = int((480 - 128) / 2)
    x1 = int((480 + 128) / 2)
    y0 = int((640 - 128) / 2)
    y1 = int((640 + 128) / 2)
    frame = frame[x0 : x1, y0 : y1]
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = 255 - frame
        in_frame = numpy.array(frame, numpy.float).reshape(1, 128, 128)
        prediction = model.predict(in_frame)[0] 
        print(f'prediction is {prediction}')
        plt.imshow(frame, 'gray')
        plt.show()
    elif k%256 == 48:
        path =r'C:\MyProjects\Python\TensorFlow\Raw DataSet\test\ffb7e43c-deac-4b8f-83bf-44437104c35d_3R.png'
        frame = cv2.imread(path, 0)
        in_frame = numpy.array(frame).reshape(1, 128, 128)
        print(model.predict(in_frame)[0])
        plt.imshow(frame, 'gray')
        plt.show()
        # cropped = frame[70:170, 440:540]
        # # SPACE pressed
        #img_name = "opencv_frame_{}.png".format(img_counter)
        # cv2.imwrite(img_name, frame)
        # print("{} written!".format(img_name))
        # img_counter += 1

cam.release()

cv2.destroyAllWindows()