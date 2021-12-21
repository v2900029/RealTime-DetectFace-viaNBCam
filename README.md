# RealTime-DetectFace-viaNBCam

Real-time Face Detection using Laptop camera

Note that tensorflow-gpu version can be used instead if a GPU device is available on the system, which will speedup the results.

## Requirements

* Python 3
* tensorflow >= 1.9.0
* opencv-python == 4.5.3
* keras == 2.4.0

## USAGE

The following example illustrates the ease of use of this package:

.. code:: python

    >>> import cv2 
    >>> from datetime import datetime
    >>> from utils.detect_face import FaceDector
    >>> ## Using NBcam
    >>> open_camera(cam_id=0, cap_size=(1280,720), backend='mtcnn')
    >>> open_camera(cam_id=0, cap_size=(1280,720), backend='opencv')
    >>> ## Using .mp4 video 
    >>> open_camera(cam_id='data/2021102716333349.mp4', cap_size=(1280,720), backend='mtcnn')


## Reference

1. MTCNN (https://github.com/ipazc/mtcnn)
2. OpenCV (https://opencv.org/)
