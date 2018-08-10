#coding:utf-8
import sys
sys.path.append('/'.join(sys.path[0].split('/')[:-1]))
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
import cv2
import os
import numpy as np

pnet_model_path = '../data/MTCNN_model/PNet_landmark/PNet'
rnet_model_path = '../data/MTCNN_model/RNet_landmark/RNet'
onet_model_path = '../data/MTCNN_model/ONet_landmark/ONet'

def main():
    test_mode = "onet"
    thresh = [0.9, 0.6, 0.7]
    min_face_size = 24
    stride = 2
    slide_window = False
    shuffle = False
    #vis = True
    detectors = [None, None, None]
    prefix = [pnet_model_path, rnet_model_path, onet_model_path]
    epoch = [18, 14, 16]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet
    RNet = Detector(R_Net, 24, 1, model_path[1])
    detectors[1] = RNet
    ONet = Detector(O_Net, 48, 1, model_path[2])
    detectors[2] = ONet
    # videopath = "./video_test.avi"
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                stride=stride, threshold=thresh, slide_window=slide_window)

    video_capture = cv2.VideoCapture(0)  #捕获摄像头
    video_capture.set(3, 340)
    video_capture.set(4, 480)
    corpbbox = None
    while True:
        # fps = video_capture.get(cv2.CAP_PROP_FPS)
        t1 = cv2.getTickCount()
        ret, frame = video_capture.read()
        if ret:
            image = np.array(frame)
            boxes_c,landmarks = mtcnn_detector.detect(image)
            
            print(landmarks.shape)
            t2 = cv2.getTickCount()
            t = (t2 - t1) / cv2.getTickFrequency()
            fps = 1.0 / t
            
            # BGR 图像转换为 YCrCb = [y, Cr, Cb] 包含三个分量的数组
            yCrCb = BGR2YCrCb(image)

            #人脸区域
            for i in range(boxes_c.shape[0]):
                bbox = boxes_c[i, :4]
                score = boxes_c[i, 4]
                corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                # if score > thresh:
                cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                            (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
                cv2.putText(frame, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
            #帧率
            cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 255), 2)
            #人脸特征点区域
            for i in range(landmarks.shape[0]):
                for j in range(len(landmarks[i])/2):
                    cv2.circle(frame, (int(landmarks[i][2*j]),int(int(landmarks[i][2*j+1]))), 2, (0,0,255))  

            #耳部区域
            for i in range(boxes_c.shape[0]):
                bbox = boxes_c[i, :4]
                ear_bboxs = cal_ear_area_pt(bbox)
                for j in ear_bboxs:
                    corpbbox = [int(j[0]), int(j[1]), int(j[2]), int(j[3])]
                    cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                            (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)

            # time end
            cv2.imshow("", frame)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break
        else:
            print('device not find')
            break
    video_capture.release()
    cv2.destroyAllWindows()

#计算耳部区域
#params  face_box: 脸部矩形区域，包含左上角点坐标和右下角点坐标
def cal_ear_area_pt(face_box):
	assert len(face_box) > 3, "face box len error"

	#左上角坐标
	face_left_up_pt = face_box[:2]

	#右下角坐标
	face_right_down_pt = face_box[2:4]

	face_height = face_right_down_pt[1] - face_left_up_pt[1]
	face_width = face_right_down_pt[0] - face_left_up_pt[0]

	ear_height = 1.1 * face_height
	ear_width = 0.6 * face_width

	#左耳朵区域，左上角和右下角坐标
	ear_left_pt = []
	ear_left_pt.append(face_left_up_pt[0] - 0.5 * face_width)
	ear_left_pt.append(face_left_up_pt[1] + 0.3 * face_height)
	ear_left_pt.append(ear_left_pt[0] + 0.6 * face_width)
	ear_left_pt.append(ear_left_pt[1] + 1.1 * face_height)

	#右耳朵区域，左上角和右下角坐标
	ear_right_pt = []
	ear_right_pt.append(face_left_up_pt[0] + 0.9 * face_width)   
	ear_right_pt.append(face_left_up_pt[1] + 0.3 * face_height)
	ear_right_pt.append(ear_right_pt[0] + 0.6 * face_width)
	ear_right_pt.append(ear_right_pt[1] + 1.1 * face_height)

	return [ear_left_pt, ear_right_pt]

def BGR2YCrCb(img_bgr):
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    return cv2.split(img_ycrcb)

# # description: 检测肤色区域
# # params: 
# # img_ycrcb: YCrCb颜色空间图像  face_box: 脸部矩形区域，包含左上角点坐标和右下角点坐标  ear_boxs: 包含左右耳朵矩形区域
# def detect_skin_area(img_ycrcb, face_box, ear_boxs):
    
#     y, cr, cb = img_ycrcb

#     #左上角坐标
# 	face_left_up_pt = face_box[:2]

# 	#右下角坐标
# 	face_right_down_pt = face_box[2:4]

#     cr_face = cr[face_left_up_pt[0]:face_right_down_pt[0], face_left_up_pt[1], face_right_down_pt[1]]
#     cb_face = cb[face_left_up_pt[0]:face_right_down_pt[0], face_left_up_pt[1], face_right_down_pt[1]]

# # 计算脸部区域肤色信息期望值向量
# def cal_expect_vetor(cr, cb):

    
if __name__ == "__main__":
    main()
