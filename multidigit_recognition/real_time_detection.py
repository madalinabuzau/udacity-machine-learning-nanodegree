# Import useful libraries
from keras.models import load_model
import numpy as np
import cv2

def recognize_digits_bbox(filename='first_video.m4v', save_as = 'output'):
    # Specify font family for text
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Load best model
    model = load_model('reg_class_model.h5')
    # Open video
    cap = cv2.VideoCapture(filename)
    # Get the width and height of the video
    height = int(cap.get(4))
    width = int(cap.get(3))
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter(save_as + '_classified.avi', fourcc, 20.0, (width,height))
    while(cap.isOpened()):
        ret, frame = cap.read()
        gray_frame = frame.mean(axis=2)
        res = cv2.resize(gray_frame, (64,64),
                interpolation = cv2.INTER_AREA).reshape(1,64,64,1)
        res = res/255
        y_pred = model.predict(res)
        num_pred = np.hstack(
                    [np.argmax(y_pred[i], axis=1).reshape(-1,1) for i in range(5)])
        num_pred = num_pred[num_pred!=10]
        for k in range(0, 20, 4):
            pred_tl = (int(y_pred[5][:,k]*width),
                       int(y_pred[5][:,k+1]*height))
            pred_br = (int(y_pred[5][:,k+2]*width),
                       int(y_pred[5][:,k+3]*height))
            pred_bound = cv2.rectangle(frame, pred_tl, pred_br, (0,255,0), 5)
        cv2.putText(frame, str(num_pred)[1:-1], (10,100), font, 4, (255,255,255))
        cv2.imshow('frame',frame)
        output.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
