import numpy as np
import cv2
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)
i = 500
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y + h, x:x + w]

        if w == 228:
            i += 1
            print("Salvato" + str(i))
            roi_color = cv2.resize(roi_color, dsize=(228, 228), interpolation=cv2.INTER_CUBIC)
            img_item = str(i) + "my-image.png"
            cv2.imwrite("data\separatePhotos\Donato2/" + img_item, roi_color)
            # cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

        color = (255, 100, 50)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
