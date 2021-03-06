#coding:utf-8
import sys
# sys.path.append('/'.join(sys.path[0].split('/')[:-1]))
sys.path.append('..')
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
import cv2
import os
import numpy as np
from skimage.feature import hog 
import sklearn.svm as ssv 
from sklearn.externals import joblib  
from scipy.stats import multivariate_normal
import time
from enum import Enum


def main():
    is_set_buffer_len = False
    # MTCNN 人脸检测器
    mtcnn_detector = mtcnn_detector_init()

    # SVM 检测器
    (clf, mean, eigVects) = svm_classification_init()

     # 打电话行为检测器
    behavior_detector = behavior_detection()

    videopath = "./IMG_1680.mp4"
    video_capture = cv2.VideoCapture(videopath)  #捕获摄像头
    video_capture.set(3, 320)
    video_capture.set(4, 480)
    
    while True:
        # fps = video_capture.get(cv2.CAP_PROP_FPS)
        t3 = time.time()
        t1 = cv2.getTickCount()
        ret, frame = video_capture.read()
        if ret:
            image = np.array(frame)
         
            boxes_c,landmarks = mtcnn_detector.detect(image)
            t4 = time.time()
            # print("mtcnn_detector frame cost time:%f"%(t4-t3))

            if len(boxes_c) == 0:
                behavior_detector.detect(frame, 0)
                cv2.imshow("", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            gray = skinDetect(image, boxes_c[0], landmarks[0])
            t5 = time.time()
            # print("skinDetect frame cost time:%f"%(t5-t4))
            # HOG 
            fd = hog_feature(gray)
            
            feature = fd.reshape((1, -1))
            feature = feature - mean
            feature = feature * eigVects
            
            t2 = cv2.getTickCount()
            t = (t2 - t1) / cv2.getTickFrequency()
            fps = 1.0 / t
            if is_set_buffer_len == False:
                is_set_buffer_len = True
                behavior_detector.init_duration(fps)
            #帧率
            cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 255), 2)


            result = clf.predict(feature) 
            behavior_detector.detect(frame, int(result))
                    
            t6 = time.time()
            # print("one frame cost time:%f"%(t6-t3))
            # time end
            cv2.imshow("gray", gray)

            show_face_ear_area(frame, boxes_c, landmarks)
            cv2.imshow("", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print('device not find')
            break
    video_capture.release()
    cv2.destroyAllWindows()

##################################################
# MTCNN 人脸检测器
##################################################
def mtcnn_detector_init():

    pnet_model_path = '../data/MTCNN_model/PNet_landmark/PNet'
    rnet_model_path = '../data/MTCNN_model/RNet_landmark/RNet'
    onet_model_path = '../data/MTCNN_model/ONet_landmark/ONet'
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
    
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                stride=stride, threshold=thresh, slide_window=slide_window)

    return mtcnn_detector

##################################################
# SVM 检测器
##################################################
def svm_classification_init():
    n = 100
    m = '20pixel'  
    model_path = './models/%s/svm_%s_pca_%s.model' %(m,m,n)

    meanVal = joblib.load('./features/PCA/%s/meanVal_train_%s.mean' %(m,m)) 
    n_eigVects = joblib.load('./features/PCA/%s/n_eigVects_train_%s_%s.eig' %(m,m,n))  

    # SVM 分类器
    clf = joblib.load(model_path)

    return (clf, meanVal, n_eigVects)

##################################################
#自适应肤色检测，根据面部肤色区域，计算均值 & 协方差矩阵
#高斯模型概率密度分布
##################################################
def skinDetect(image, bbox, landmark):
    # t1 = time.time()
    #裁剪出脸部区域
    crop_image = face_image(image, bbox)
    
    #YCbCr 生成二维数组[[cb,cb,cb,cb,....], [cr,cr,cr,cr,cr,...]]
    cbcr = image_reshape(crop_image)

    #计算人脸区域期望值
    cbcr_mean = mean(cbcr)

    #计算人脸区域协方差
    cbcr_cov = covariance_matrix(cbcr)

    #计算face区域单高斯概率分布
    face_single_gaussian_model(cbcr, cbcr_mean, cbcr_cov)
    
    #获取感兴趣区域图像(face & ear)
    (position,roi_image) = ROI(image, bbox)

    # resize to 200 * 200
    roi_image = cv2.resize(roi_image,(200,200),interpolation=cv2.INTER_CUBIC) 

    #计算整张图片的单峰高斯概率密度pdf
    gray = single_gaussian_model(roi_image, position, cbcr_mean, cbcr_cov)

    t2 = time.time()
    # print("skinDetect cost time:%f"%(t2 - t1))
    return gray

##################################################
#获取图像Hog特征
##################################################
def hog_feature(image):
    visualize = False  
    block_norm = 'L2-Hys'  
    cells_per_block = (2,2) 
    pixels_per_cell = (20,20)  
    orientations = 9  

    fd = hog(image, orientations, pixels_per_cell, cells_per_block, block_norm, visualize)
    return fd

##################################################
#HOG 特征采用PCA降维
##################################################
def pca_feature(features, dimension = 100):
    print("Start to do PCA...")   
 
    t1 = time.time()   
    newData,meanVal=zeroMean(features)   
    covMat=np.cov(newData,rowvar=0)   
    eigVals,eigVects=np.linalg.eig(np.mat(covMat)) # calculate feature value and feature vector   
    
    joblib.dump(eigVals,'./features/PCA/%s/eigVals_train_%s.eig' %(m,m),compress=3)    
    joblib.dump(eigVects,'./features/PCA/%s/eigVects_train_%s.eig' %(m,m),compress=3)  

    eigValIndice=np.argsort(eigVals) # sort feature value
    n_eigValIndice=eigValIndice[-1:-(dimension+1):-1] # take n feature value   
    n_eigVect=eigVects[:,n_eigValIndice] # take n feature vector 
    joblib.dump(n_eigVect,'./features/PCA/%s/n_eigVects_train_%s_%s.eig' %(m,m,dimension))    
    lowDDataMat=newData*n_eigVect # calculate low dimention data
     
    t2 = time.time()   
    print("PCA takes %f seconds" %(t2-t1))   
    
    return lowDDataMat 

##################################################
# PCA降维之后的HOG特征作为输入，采用SVM分类
##################################################
def svm_classification(features, labels):
    t0 = time.time()

    model_path = './models/%s/svm_%s_pca_%s.model' %(m,m,dimension)
    clf = ssv.SVC(kernel='linear')   
    print("Training a SVM Classifier.")  
    print("fds type: %s  shape:%s"%(type(features), features.shape))
    # print("lable type: %s  shape:%s"%(type(labels), labels.shape))
    clf.fit(features, labels)   
    joblib.dump(clf, model_path)  
  
    t1 = time.time()   
    print("Classifier saved to {}".format(model_path))
    print('The cast of time is :%f seconds' % (t1-t0))


def zeroMean(dataMat): # zero normalisation
    meanVal=np.mean(dataMat,axis=0) # calculate mean value of every column.   
    joblib.dump(meanVal,'./features/PCA/%s/meanVal_train_%s.mean' %(m,m)) # save mean value   
    newData=dataMat-meanVal   
    return newData,meanVal

##################################################
#采用mean公式计算均值
##################################################
def mean(cbcr):
    cbcr_mean = np.mean(cbcr, axis=1)
    # print("cbcr_mean:", cbcr_mean)
    return cbcr_mean

##################################################
# 采用cov公式计算协方差
##################################################
def covariance_matrix(cbcr):
    cbcr_cov = np.cov(cbcr)
    # print("cbcr_cov:", cbcr_cov)
    return cbcr_cov


def face_single_gaussian_model(cbcr, mean, cov):
    # t1 = time.time()
    #---生成数组[[cb,cr], [cb,cr] ... [cb,cr] ]
    cbcr = np.column_stack((cbcr[0], cbcr[1]))

    pdfs = multivariate_normal.pdf(cbcr, mean = mean, cov=cov, allow_singular=True)

    global possibly
    possibly = find_gaussion_probability_threshold(pdfs)
    t2 = time.time()
    # print("face_single_gaussian_model cost time:%f"%(t2-t1)) 

def single_gaussian_model(image, position, mean, cov):
    
    #转换为YCbCr 且生成二维数组[[cb,cb,cb,cb,....], [cr,cr,cr,cr,cr,...]]
    cbcr = image_reshape(image)
    cbcr = np.column_stack((cbcr[0], cbcr[1]))

    #计算高斯概率密度分布
    pdfs = multivariate_normal.pdf(cbcr, mean = mean, cov=cov, allow_singular=True)
    
    pdfs[pdfs >= possibly] = 255    #矩阵元素大于阈值设置为白色
    pdfs[pdfs < possibly] = 0       #矩阵小雨阈值设置为黑色

    img_gray = pdfs.reshape(image.shape[0], image.shape[1])
    
    img_gray = close_operation(img_gray)
    # cv2.imshow("gray",img_gray)
    # cv2.waitKey(0) & 0xFF == ord('q')
    return  img_gray  

def find_gaussion_probability_threshold(pdfs):
    # t1 = time.time()
    pdfs.sort()
    length = len(pdfs)
    right = len(pdfs) - 1
    left = 0
    while left <= right:
        mid = int((right + left) / 2)
        rate = float(len(pdfs[pdfs >= pdfs[mid]])) / length
        if rate >= 0.7 and rate <= 0.71:
            break
        else:
            if rate < 0.7:
                right = mid -1
            else:
                left = mid + 1
    t2 = time.time()
    # print("find_gaussion_probability_threshold cost time:%f"%(t2-t1))        
    return pdfs[mid]

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
	ear_width = 1.0 * face_width

	#左耳朵区域，左上角和右下角坐标
	ear_left_pt = []
	ear_left_pt.append(face_left_up_pt[0] - 0.7 * face_width)
	ear_left_pt.append(face_left_up_pt[1] + 0.3 * face_height)
	ear_left_pt.append(ear_left_pt[0] + 0.8 * face_width)
	ear_left_pt.append(ear_left_pt[1] + 1.1 * face_height)

	#右耳朵区域，左上角和右下角坐标
	ear_right_pt = []
	ear_right_pt.append(face_left_up_pt[0] + 0.9 * face_width)   
	ear_right_pt.append(face_left_up_pt[1] + 0.3 * face_height)
	ear_right_pt.append(ear_right_pt[0] + 0.8 * face_width)
	ear_right_pt.append(ear_right_pt[1] + 1.1 * face_height)

	return [ear_left_pt, ear_right_pt]

#感兴趣区域图像截取(face & ear)
def ROI(image, bbox):
    ear_bboxs = cal_ear_area_pt(bbox)

    left_ear = ear_bboxs[0]
    right_ear = ear_bboxs[1]

    #赋值的同时判断是否超出图像边界
    left_point_x = int(round(left_ear[0])) if int(round(left_ear[0])) > 0 else 0
    left_point_y = int(round(bbox[1])) if int(round(bbox[1])) > 0 else 0

    right_point_x = int(round(right_ear[2])) if int(round(right_ear[2])) < image.shape[1] else image.shape[1]
    right_point_y = int(round(right_ear[3])) if int(round(right_ear[3])) < image.shape[0] else image.shape[0]

    img = image[left_point_y:right_point_y, left_point_x : right_point_x] 

    x1 = bbox[0] - left_point_x #人脸左上角新位置x
    y1 = left_ear[1] - bbox[1]  #左耳朵左上角新位置y
    x2 = bbox[2] - left_point_x 
    y2 = right_ear[1] - bbox[1] 

    # cv2.rectangle(img, (0,0), (int(x1), int(y1)), (0,0,255))
    # cv2.rectangle(img, (int(x2),0), (img.shape[1], int(y2)), (0,0,255))
    # print("img shape:",img.shape)
    # cv2.imshow("ROI",img)
    # cv2.waitKey(0) & 0xFF == ord('q')
    return ([x1,y1,x2,y2],img)

def face_image(image, bbox):
    y1 = int(round(bbox[1])) if int(round(bbox[1])) > 0 else 0
    y2 = int(round(bbox[3])) if int(round(bbox[3])) > 0 else 0
    x1 = int(round(bbox[0])) if int(round(bbox[0])) > 0 else 0
    x2 = int(round(bbox[2])) if int(round(bbox[2])) > 0 else 0

    crop_image = image[y1:y2, x1:x2]
    return crop_image

# 生成二维数组[[cb,cb,cb,cb,....], [cr,cr,cr,cr,cr,...]]
def image_reshape(image):
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    _, cr, cb = cv2.split(img_ycrcb)
    crravel = cr.ravel()
    cbravel = cb.ravel()
    cbcr = np.vstack((cbravel, crravel))
    return cbcr


def BGR2YCrCb(img_bgr):
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)

    return cv2.split(img_ycrcb)

# 闭运算
def close_operation(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return img

# 开运算
def open_operation(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return img


class behavior_detection(object):
    LaunchModeDuration = 30        #开始模式缓存帧数
    RealtimeModeDuration = 8       #实时模式
    ExitModeDuration = 8           #退出模式

    LaunchAlarmThreshold = 0.60
    RealAlarmThreshold = 0.55
    ExitAlarmThreshold = 0.50

    def __init__(self):
        self.state = RunState.Launch
        self.buffer = [[],[],[]]
        self.index = 0
        self.isMakingPhone = False

    def init_duration(self, fps):
        self.LaunchModeDuration = int(2.5 * fps)
        self.RealtimeModeDuration = int(1 * fps)
        self.ExitModeDuration = int(1 * fps)

    def detect(self, frame, value):
        launchBuffer = self.buffer[0]        #开始模式缓存Frame判别结果Buffer
        realTimeBuffer = self.buffer[1]      #实时模式
        exitBuffer = self.buffer[2]          #退出模式
        isStateChanged = False

        self.isMakingPhone = False
        duration = [self.LaunchModeDuration, self.RealtimeModeDuration, self.ExitModeDuration]
        if self.state == RunState.Launch: #开始模式
            if self.index >= len(launchBuffer):
                launchBuffer.append(value)
            else:
                launchBuffer[self.index] = value
            
            if len(launchBuffer) == duration[self.state.value]:
                if (float(sum(launchBuffer)) / duration[self.state.value]) >= self.LaunchAlarmThreshold:
                    self.isMakingPhone = True
                    isStateChanged = True
                    self.state = RunState.Real
                    self.index = 0
                    del realTimeBuffer[:]

            print("launchBuffer len:%d  sum:%d"%(len(launchBuffer), sum(launchBuffer)))
        elif self.state == RunState.Real:
            self.isMakingPhone = True
            if self.index >= len(realTimeBuffer):
                realTimeBuffer.append(value)
            else:
                realTimeBuffer[self.index] = value
            
            if len(realTimeBuffer) == duration[self.state.value]:
                if (float(sum(realTimeBuffer)) / duration[self.state.value]) < self.RealAlarmThreshold:
                    isStateChanged = True
                    self.state = RunState.Exit
                    self.index = 0
                    del exitBuffer[:]
            print("realTimeBuffer len:%d  sum:%d"%(len(realTimeBuffer), sum(realTimeBuffer)))
        elif self.state == RunState.Exit:
            self.isMakingPhone = True 
            if self.index >= len(exitBuffer):
                exitBuffer.append(value)
            else:
                exitBuffer[self.index] = value

            if len(exitBuffer) == duration[self.state.value]:
                isStateChanged = True
                if (float(sum(exitBuffer)) / duration[self.state.value])  >= self.ExitAlarmThreshold:
                    self.state = RunState.Real
                    self.index = 0
                    del realTimeBuffer[:]
                else:
                    self.isMakingPhone = False 
                    self.state = RunState.Launch
                    self.index = 0
                    del launchBuffer[:]
            print("exitBuffer len:%d  sum:%d"%(len(exitBuffer), sum(exitBuffer)))
        if isStateChanged == False:
            self.index += 1
            self.index = self.index % duration[self.state.value]

        print("current index:%d  value:%d  isMakingPhone:%d "%(self.index, value, self.isMakingPhone))

        image = np.array(frame)
        if self.isMakingPhone == True:
            cv2.circle(frame, (image.shape[1] - 20,20), 10, (0,0,255), -1)
        else:
            cv2.circle(frame, (image.shape[1] - 20,20), 10, (0,255,0), -1)

class RunState(Enum):
    Launch = 0      #开始模式
    Real = 1        #实时模式
    Exit = 2        #退出模式

def show_face_ear_area(frame, boxes_c, landmarks):

    # t2 = cv2.getTickCount()
    # t = (t2 - t1) / cv2.getTickFrequency()
    # fps = 1.0 / t

    #人脸区域
    for i in range(boxes_c.shape[0]):
        bbox = boxes_c[i, :4]
        score = boxes_c[i, 4]
        corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        # if score > thresh:
        cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                    (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
        cv2.putText(frame, 'face', (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 1)
    #帧率
    # cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             (255, 0, 255), 2)
    #人脸特征点区域
    for i in range(landmarks.shape[0]):
        for j in range(int(len(landmarks[i])/2)):
            cv2.circle(frame, (int(landmarks[i][2*j] + 5),int(landmarks[i][2*j+1] + 3)), 2, (0,0,255))  

    #耳部区域
    for i in range(boxes_c.shape[0]):
        bbox = boxes_c[i, :4]
        ear_bboxs = cal_ear_area_pt(bbox)
        for j in ear_bboxs:
            corpbbox = [int(j[0]), int(j[1]), int(j[2]), int(j[3])]
            cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                    (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)


    
if __name__ == "__main__":
    main()
