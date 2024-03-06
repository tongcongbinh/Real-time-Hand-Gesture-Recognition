import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import pickle
#Lay du lieu diem anh

# initialize mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence=0.3)


DATA_FOLDER = '.\data'

data = []
labels = []


for folder_ in os.listdir(DATA_FOLDER): # so luong folder anh
    
    for img_path in os.listdir(os.path.join(DATA_FOLDER, folder_)): # so luong anh
        data_aux = []
        
        img = cv2.imread(os.path.join(DATA_FOLDER, folder_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:                
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
            data.append(data_aux)
            labels.append(folder_)
f = open('.\data.pickle', 'wb')
pickle.dump({'data': data, 'lables':labels}, f)
f.close()


'''      
        plt.figure()
        plt.imshow(img_rgb)

plt.show()
'''          