import numpy as np
import cv2 
from mtcnn.mtcnn import MTCNN
import logging

class FaceDector(object):
    def __init__(self, backend, opencv_xml_path=None, score_thr=0.90):
        self.backend = backend.lower()
        self.opencv_xml_path = opencv_xml_path
        self.face_dector = self.__prepare_detector()
        self.score_thr = score_thr
        # self.support_backends = ['opencv', 'mtcnn']

    def __prepare_detector(self):
        if self.backend == 'opencv':
            face_dector = cv2.CascadeClassifier(self.opencv_xml_path)
        elif self.backend == 'mtcnn':
            face_dector = MTCNN()
        return face_dector

    def __encode_detection(self, detections):
        faces = []
        if self.backend == 'opencv':
            for (x, y, w, h) in detections:
                face = {'box': [x, y, w, h], 'backend': self.backend, 'keypoints': None, 'confidence': None}
                faces.append(face)
        elif self.backend == 'mtcnn':
            for face in detections:
                face['backend'] = self.backend
                faces.append(face)
        else:
            faces = None
        return faces

    def _update_log(self, face_index, detection, filename, verbose=0):
        x, y, w, h = detection["box"]
        if self.backend =='opencv' and verbose == 1:
            logging.info("--------------------------------------------")
            logging.info("Face {}{}:".format(filename, face_index))
            logging.info("\tbbox: (x, y, w, h) = ({0}, {1}, {2}, {3})".format(x, y, w, h))
        elif self.backend =='mtcnn' and verbose == 1:
            logging.info("--------------------------------------------")
            logging.info("Face {}{}:".format(filename, face_index))
            logging.info("\tbbox: (x, y, w, h) = ({0}, {1}, {2}, {3})".format(x, y, w, h))
            logging.info("\tconfidence: {:.3f}".format(detection["confidence"]))     

    def detecFace(self, frame, scale_factor=1.2, min_neighbors=5, min_size=(50,50)):
        if self.backend == 'opencv':
            input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # RGB to gray image
            detections = self.face_dector.detectMultiScale(input_img,
                                                        scaleFactor = scale_factor,
                                                        minNeighbors = min_neighbors,
                                                        minSize = min_size,
                                                        flags = cv2.CASCADE_SCALE_IMAGE)
            faces = self.__encode_detection(detections)
        elif self.backend == 'mtcnn':
            detections = self.face_dector.detect_faces(frame)
            faces = self.__encode_detection(detections)
        
        return faces                

    def save_faces(self, frame, faces, save_name='output', save_path='./', verbose=1):
        if self.backend == 'opencv':
            face_index=0
            for detection in faces:
                face_index += 1
                x, y, w, h = detection["box"]
                detected_face = frame[int(y):int(y+h), int(x):int(x+w)]
                cv2.imwrite('{}{}{:3d}.jpg'.format(save_path, save_name, face_index), detected_face)
                self._update_log(face_index=face_index, detection=detection, filename=save_name, verbose=verbose)

        elif self.backend == 'mtcnn':
            for detection in faces:
                if detection["confidence"]  >= self.score_thr:
                    face_index += 1
                    x, y, w, h = detection["box"]
                    detected_face = frame[int(y):int(y+h), int(x):int(x+w)]
                    cv2.imwrite('{}{}{:3d}.jpg'.format(save_path, save_name, face_index), detected_face)
                    self._update_log(face_index=face_index, detection=detection, filename=save_name, verbose=verbose)

    def draw_faces(self, frame, faces, fontScale=0.5, lineColor=(0,255,255)):
        # Draw a rectangle around the faces
        for detections in faces:
            # get coordinates
            (x, y, w, h) = detections['box']
            cv2.rectangle(frame, (x, y), (x+w, y+h), lineColor, 2)
            if self.backend == 'mtcnn':
                wh_text = 'Face: ({0},{1}), score={2:.2f}'.format(w,h,detections['confidence'])
            else:
                wh_text = 'Face: ({0},{1}), score={2}'.format(w,h,detections['confidence'])
            cv2.putText(frame, wh_text, (x+w, y), cv2.FONT_HERSHEY_COMPLEX_SMALL,  fontScale, lineColor, 1, cv2.LINE_AA)
            # cv2.imshow('frame', frame)
        return frame

