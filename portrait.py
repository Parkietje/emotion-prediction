import cv2
import time
import os
import uuid

OUTPUT_DIRECTORY = "screenshots"

def capture():
    cap = cv2.VideoCapture(0)
    
    # wait until camera is ready
    time.sleep(0.2)
    
    # Capture frame
    _, frame = cap.read()
    
    # crop face
    frame = _crop_face(frame)
    if frame is None:
        raise Exception('No face found')

    # write to file
    id = uuid.uuid1()
    image_name = str(id) + ".jpg"
    path = os.path.join(OUTPUT_DIRECTORY, image_name)
    cv2.imwrite(path, frame)

    # When everything done, release the capture
    cap.release()    
    return path

def _crop_face(img):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    count = 0
    for (x,y,w,h) in faces:
        # return first face in img
        return img[y:y+h, x:x+w]

if __name__ == '__main__':
    capture()