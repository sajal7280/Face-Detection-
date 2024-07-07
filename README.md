
import cv2
import os

cascade_filename = 'haarcascade_frontalface_default.xml'
cascade_path = os.path.join(cv2.data.haarcascades, cascade_filename)

cascade = cv2.CascadeClassifier(cascade_path)

if cascade.empty():
    raise IOError(f"Unable to load the cascade classifier XML file: {cascade_filename}")

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()

    if not ret:
        print("Failed to grab frame")
        break

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Face Detection", img)

    key = cv2.waitKey(1)

    if key == 81 or key == 113:  # 81 is ASCII for 'Q' and 113 is ASCII for 'q'
        print("Stopping the camera")
        break

cam.release()

cv2.destroyAllWindows()
