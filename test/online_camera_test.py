#coding:utf-8
import sys
sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
import cv2
import numpy as np
import os

test_mode = "onet"
thresh = [0.9, 0.6, 0.7]
min_face_size = 24
stride = 2
slide_window = False
shuffle = False
#vis = True
detectors = [None, None, None]
prefix = ['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet', '../data/MTCNN_model/ONet_landmark/ONet']
epoch = [18, 14, 16]
model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet
RNet = Detector(R_Net, 24, 1, model_path[1])
detectors[1] = RNet
ONet = Detector(O_Net, 48, 1, model_path[2])
detectors[2] = ONet
videopath = "./video"
# mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
#                                stride=stride, threshold=thresh, slide_window=slide_window)

paths = os.listdir(videopath)
num = 0
# paths = ["13025123.mp4", "IMG_1679.mp4", "IMG_1680.mp4", "IMG_1681.mp4"]
for path in paths:
    if path.endswith('.mp4') == False:
        continue
    filepath = os.path.join(videopath,path)
    
    name = path.split('.')
    save_path = os.path.join(videopath,name[0])
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)

    video_capture = cv2.VideoCapture(filepath)
    if video_capture.isOpened == False:
        continue
    video_capture.set(3, 340)
    video_capture.set(4, 480)
    
    print("Start to write video %s"%(path))
    while True:
        # fps = video_capture.get(cv2.CAP_PROP_FPS)
        t1 = cv2.getTickCount()
        ret, frame = video_capture.read()
        if ret:
            image = np.array(frame)

            if num % 10 == 0:
                cv2.imwrite("%s/img_%d.jpg"%(save_path,num), image) 
            num += 1
            # time end
            cv2.imshow("", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print 'device not find'
            break
    video_capture.release()
    cv2.destroyAllWindows()
