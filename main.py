import numpy as np
import cv2 
from datetime import datetime
from utils.detect_face import FaceDector
import logging

logging.basicConfig(level=logging.INFO, filename='logs/{}.txt'.format(datetime.now().strftime('%Y%m%d')), filemode='w',
	                    format='[%(asctime)s %(levelname)-8s] %(message)s',
	                    datefmt='%Y%m%d %H:%M:%S',)


def open_camera(cam_id=0, cap_size=(1280,720), backend='mtcnn'):
    camera_name='output'
    # 選擇第二隻攝影機
    cap = cv2.VideoCapture(cam_id)
    # 設定影像的尺寸大小
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_size[1])

    if backend =='opencv':
        face_detector = FaceDector(backend=backend, opencv_xml_path='weights/haarcascade_frontalface_default.xml')
    elif backend == 'mtcnn':
        face_detector = FaceDector(backend=backend, score_thr=0.90)

    while(True):
        # 從攝影機擷取一張影像
        ret, frame = cap.read()
        if ret:
            # timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            faces = face_detector.detecFace(frame)
            face_detector.save_faces(frame, faces, save_name='{0}-{1}'.format(camera_name, datetime.now().strftime('%Y%m%d %H-%M-%S.%f')[:-3]), save_path='./outputs/', verbose=1)
            frame = face_detector.draw_faces(frame, faces, fontScale=0.5, lineColor=(0,255,255))
            # show cap frame with fps
            face_detector.update_fps()
            face_detector.show_image(frame, with_fps=True, fontScale=0.5, lineColor=(0,255,255))

        if face_detector.fps_idx>=1024:
            face_detector.start_time = datetime.now()
            face_detector.fps_idx=0

        # 若按下 q 鍵則離開迴圈
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   

    # Release the camera 
    cap.release()
    # Close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == '__main__':

    open_camera(cam_id=0, cap_size=(1280,720), backend='mtcnn')
    # open_camera(cam_id=0, cap_size=(1280,720), backend='opencv')