import pickle
import cv2
import numpy as np
import mediapipe as mp


model_dict = pickle.load(open('.\HandGestureRecognition\model.p','rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(max_num_hands=2,static_image_mode = True, min_detection_confidence=0.3)

labels_dict = {'okay':'okay', 'peace':'peace', 'thumbs up':'thumbs up', 'thumbs down':'thumbs down','call me':'call me',
                'stop':'stop', 'rock':'rock', 'live long':'live long', 'fist':'fist', 'smile':'smile'}
while True:
    data_aux = []
    x_ = []
    y_ = []
    # Read each frame from the webcam
    _, frame = cap.read()
    H , W, __ = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    gesture_name = ''
    position_x = 0
    position_y = 0
        
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                    frame, #image to draw
                    hand_landmarks, #model output
                    mp_hands.HAND_CONNECTIONS, #hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )


        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)
        
        
        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        
        x2 = int(max(x_) * W) + 10
        y2 = int(max(y_) * H) + 10
        
        prediction = model.predict([np.asarray(data_aux)])
        predicted_charater = labels_dict[prediction[0]]
        gesture_name = predicted_charater
        position_x = x1
        position_y = y1
        cv2.rectangle(frame, (x1, y1), (x2,y2), (148,238,130), 4)
    
    cv2.putText(frame, gesture_name, (position_x, position_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (165,120,25), 2, cv2.LINE_AA)
    
    cv2.imshow("Output", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()