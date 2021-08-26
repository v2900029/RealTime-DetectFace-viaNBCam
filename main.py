import numpy as np
import cv2 
import time
from mtcnn.mtcnn import MTCNN

def detectFace_from_opencv(cap):
    # init detect Face Model
    face_cascade = cv2.CascadeClassifier('D:\python_env/face_env/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    
    fontScale=0.5
    fps_idx=0
    # Start time
    start_time = time.time()
    while(True):
        fps_idx+=1
        # 從攝影機擷取一張影像
        ret, frame = cap.read()
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray_img,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            wh_text = '({0},{1})'.format(w,h)
            cv2.putText(frame, wh_text, (x+w, y), cv2.FONT_HERSHEY_COMPLEX_SMALL,  fontScale, (0, 255, 255), 1, cv2.LINE_AA)

        # Calculate frames per second
        fps  = np.int(fps_idx / (time.time() - start_time))
        fps_text = 'FPS: {0}'.format(fps)
        cv2.putText(frame, fps_text, (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,  fontScale, (0, 255, 255), 1, cv2.LINE_AA)

        # 顯示圖片
        cv2.imshow('frame', frame)

        if fps_idx>=1024:
            start_time = time.time()
            fps_idx=0

        # 若按下 q 鍵則離開迴圈
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def detectFace_form_mtcnn(cap):
    # init detect Face Model
    # model = MTCNN(weights_file='filename.npy')
    detector = MTCNN()

    fontScale=0.5
    fps_idx=0
    # Start time
    start_time = time.time()
    while(True):
        fps_idx+=1
        # 從攝影機擷取一張影像
        ret, frame = cap.read()

        faces = detector.detect_faces(frame)

        # Draw a rectangle around the faces
        for result in faces:
            # get coordinates
            (x, y, w, h) = result['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            wh_text = '({0},{1}), score={2:.2f}'.format(w,h,result['confidence'])
            cv2.putText(frame, wh_text, (x+w, y), cv2.FONT_HERSHEY_COMPLEX_SMALL,  fontScale, (0, 255, 255), 1, cv2.LINE_AA)

        # Calculate frames per second
        fps  = np.int(fps_idx / (time.time() - start_time))
        fps_text = 'FPS: {0}'.format(fps)
        cv2.putText(frame, fps_text, (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,  fontScale, (0, 255, 255), 1, cv2.LINE_AA)

        # 顯示圖片
        cv2.imshow('frame', frame)

        if fps_idx>=1024:
            start_time = time.time()
            fps_idx=0

        # 若按下 q 鍵則離開迴圈
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def open_camera():
    # # init detect Face Model
    # face_cascade = cv2.CascadeClassifier('D:\python_env/face_env/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    
    # 選擇第二隻攝影機
    cap = cv2.VideoCapture(1)
 
    # 兩種偵測人臉的方式
    # detectFace_from_opencv(cap)
    detectFace_form_mtcnn(cap)
 
    # 釋放攝影機
    cap.release()
    # 關閉所有 OpenCV 視窗
    cv2.destroyAllWindows()


if __name__ == '__main__':
    open_camera()