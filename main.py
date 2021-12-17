import numpy as np
import cv2 
import time
from utils.detect_face import FaceDector


def open_camera(cam_id=0, cap_size=(1024,768)):
    # 選擇第二隻攝影機
    cap = cv2.VideoCapture(cam_id)
    # 設定影像的尺寸大小
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_size[1])

    fps_idx=0
    face_detector = FaceDector(backend='mtcnn', score_thr=0.90)
    # face_detector = FaceDector(backend='opencv', opencv_xml_path='D:\python_env/face_env/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

    start_time = time.time()
    while(True):
        fps_idx+=1
        # 從攝影機擷取一張影像
        ret, frame = cap.read()
        if ret:
            faces = face_detector.detectFace_form_mtcnn(frame)
            # face_detector.save_faces(self, frame, faces, save_name='output', save_path='./', verbose=1)
            frame = face_detector.draw_faces(frame, faces, fontScale=0.5, lineColor=(0,255,255))

        # Calculate frames per second
        fps  = np.int(fps_idx / (time.time() - start_time))
        fps_text = 'FPS: {0}'.format(fps)
        cv2.putText(frame, fps_text, (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,  0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        if fps_idx>=1024:
            start_time = time.time()
            fps_idx=0

        # 若按下 q 鍵則離開迴圈
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   

    # 釋放攝影機
    cap.release()
    # 關閉所有 OpenCV 視窗
    cv2.destroyAllWindows()


if __name__ == '__main__':
    open_camera(cam_id=0, cap_size=(1024,768))