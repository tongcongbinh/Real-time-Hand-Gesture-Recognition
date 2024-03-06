import os

import cv2


DATA_DIR = '\HandGestureRecognition\data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

gesture = ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']
dataset_size = 88

cap = cv2.VideoCapture(0)

for j in gesture:
    if not os.path.exists(os.path.join(DATA_DIR, str(j))):
        os.makedirs(os.path.join(DATA_DIR, str(j)))

    print('Collecting data for class "{}"'.format(j))

    #done = False
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, 'Press "Q" to start collecting!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (155,231,20), 2,cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(100) == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow('frame', frame)
        cv2.waitKey(150)
        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

        counter += 1

cap.release()
cv2.destroyAllWindows()