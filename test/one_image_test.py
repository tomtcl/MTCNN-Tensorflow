#coding:utf-8
import sys
parent_path = sys.path[0].replace("/test", "", 1)
sys.path.append(parent_path)
from Detection.MtcnnDetector import MtcnnDetector
from Detection.detector import Detector
from Detection.fcn_detector import FcnDetector
from train_models.mtcnn_model import P_Net, R_Net, O_Net
from prepare_data.loader import TestLoader
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from skimage.feature import hog 
import sklearn.svm as ssv 
from sklearn.externals import joblib  
from scipy.stats import multivariate_normal

# np.set_printoptions(threshold='nan') 

possibly = 0.0      #高斯概率密度返回波峰阈值
m = '20pixel'       # HoG choses pixel per cell will gain different number of feature values. 
dimension = 100     # this is to define how dimentions u want for PCA

possibly = 0.0

def main():
    # test_svm_classification()
    paths = ["sample"]
    count = 0
    fds = []
    labels = []

    # MTCNN检测器
    mtcnn_detector = mtcnn_detector_init()    
    t0 = time.time()
    num = 0
    for path in paths:
        for item in os.listdir(path):
            if item[0] == '.':
                continue
            t1 = time.time()
            imagepath = os.path.join(path,item)
            print("%d Dealing with %s"%(num,imagepath))

            test_data = TestLoader([imagepath])
            all_boxes,landmarks = mtcnn_detector.detect_face(test_data)
            if len(all_boxes[0]) == 0:
                continue
            
            t2 = time.time()
            print("mtcnn_detector cost time %f"%(t2-t1))
            
            image = cv2.imread(imagepath)

            # img = crop_image(image, all_boxes[0][0])
            #Skin Detection
            # gray = skinDetect(image, all_boxes[count][0], landmarks[count][0])
            
            
            # HOG 
            # fd = hog_feature(gray)
            # fds.append(fd)
            # if cmp(path, "positive") == 0:
            #     labels.append(1.0)
            # else:
            #     labels.append(0.0)
            
            # t3 = time.time()
            # print("one image cost time:%f"%(t3 - t1))

            num += 1
            test_show_face_area(image, all_boxes)
            test_show_landmark_area(image, landmarks)
            cv2.imwrite("resize/img_2000%d.jpg"%(num), image)
            # cv2.imshow("lala",image)
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     continue


    # PCA 降维 
    # print("fds shape")
    # fds = np.array(fds)
    # print("type: %s  shape:%s"%(type(fds), fds.shape))
    # print(fds)
    # fds = pca_feature(fds)

    # # SVM Classification  
    # svm_classification(fds, labels)
    # t4 = time.time()
    # print("total cost time:%f"%(t4 - t0))


##################################################
# MTCNN 人脸检测器
##################################################
def mtcnn_detector_init():
    test_mode = "ONet"
    thresh = [0.9, 0.6, 0.7]
    min_face_size = 24
    stride = 2
    slide_window = False
    shuffle = False
    detectors = [None, None, None]
    prefix = ['../data/MTCNN_model/PNet_landmark/PNet', '../data/MTCNN_model/RNet_landmark/RNet', '../data/MTCNN_model/ONet_landmark/ONet']
    epoch = [18, 14, 16]
    batch_size = [2048, 256, 16]
    model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
    # load pnet model
    if slide_window:
        PNet = Detector(P_Net, 12, batch_size[0], model_path[0])
    else:
        PNet = FcnDetector(P_Net, model_path[0])
    detectors[0] = PNet

    # load rnet model
    if test_mode in ["RNet", "ONet"]:
        RNet = Detector(R_Net, 24, batch_size[1], model_path[1])
        detectors[1] = RNet

    # load onet model
    if test_mode == "ONet":
        ONet = Detector(O_Net, 48, batch_size[2], model_path[2])
        detectors[2] = ONet

    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                stride=stride, threshold=thresh, slide_window=slide_window)
    
    return mtcnn_detector

##################################################
#自适应肤色检测，根据面部肤色区域，计算均值 & 协方差矩阵
#高斯模型概率密度分布
##################################################
def skinDetect(image, bbox, landmark):
    t1 = time.time()
    #裁剪出脸部区域
    crop_image = face_image(image, bbox)
    
    #YCbCr 生成二维数组[[cb,cb,cb,cb,....], [cr,cr,cr,cr,cr,...]]
    cbcr = image_reshape(crop_image)
    # test_show_skin_piexl(cbcr[0], cbcr[1], 'b')
    #调整五个特征点相对人脸的位置(之前是相对裁剪之前图片的位置)
    # landmark = landmark_relative(bbox, landmark)

    #去除噪声(五官,内接圆外)
    # cbcr = remove_noise(crop_image, landmark, cbcr)

    #计算人脸区域期望值
    cbcr_mean = mean(cbcr)

    #计算人脸区域协方差
    cbcr_cov = covariance_matrix(cbcr)

    #计算face区域单高斯概率分布
    face_single_gaussian_model(cbcr, cbcr_mean, cbcr_cov)
    
    #获取感兴趣区域图像(face & ear)
    (position,roi_image) = ROI(image, bbox)

    # print("roi_image shape:", roi_image.shape)
    # resize to 200 * 200
    roi_image = cv2.resize(roi_image,(200,200),interpolation=cv2.INTER_CUBIC) 

    #计算整张图片的单峰高斯概率密度pdf
    gray = single_gaussian_model(roi_image, position, cbcr_mean, cbcr_cov)

    t2 = time.time()
    print("skinDetect cost time:%f"%(t2 - t1))
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
    print(fd)

    return fd

##################################################
#HOG 特征采用PCA降维
##################################################
def pca_feature(features, dimension = 100):
    print "Start to do PCA..."   
 
    t1 = time.time()   
    newData,meanVal=zeroMean(features)   
    covMat=np.cov(newData,rowvar=0)   
    eigVals,eigVects=np.linalg.eig(np.mat(covMat)) # calculate feature value and feature vector   
    
    # joblib.dump(eigVals,'./features/PCA/%s/eigVals_train_%s.eig' %(m,m),compress=3)    
    # joblib.dump(eigVects,'./features/PCA/%s/eigVects_train_%s.eig' %(m,m),compress=3)  

    eigValIndice=np.argsort(eigVals) # sort feature value
    n_eigValIndice=eigValIndice[-1:-(dimension+1):-1] # take n feature value   
    n_eigVect=eigVects[:,n_eigValIndice] # take n feature vector 
    joblib.dump(n_eigVect,'./features/PCA/%s/n_eigVects_train_%s_%s.eig' %(m,m,dimension))    
    lowDDataMat=newData*n_eigVect # calculate low dimention data
     
    t2 = time.time()   
    print "PCA takes %f seconds" %(t2-t1)   
    
    return lowDDataMat 

##################################################
# PCA降维之后的HOG特征作为输入，采用SVM分类
##################################################
def svm_classification(features, labels):
    t0 = time.time()

    model_path = './models/%s/svm_%s_pca_%s.model' %(m,m,dimension)
    clf = ssv.SVC(kernel='linear')   
    print "Training a SVM Classifier."   
    print("fds type: %s  shape:%s"%(type(features), features.shape))
    # print("lable type: %s  shape:%s"%(type(labels), labels.shape))
    clf.fit(features, labels)   
    joblib.dump(clf, model_path)  
  
    t1 = time.time()   
    print "Classifier saved to {}".format(model_path)   
    print 'The cast of time is :%f seconds' % (t1-t0) 


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
    print("cbcr_mean:", cbcr_mean)
    return cbcr_mean

##################################################
# 采用cov公式计算协方差
##################################################
def covariance_matrix(cbcr):
    cbcr_cov = np.cov(cbcr)
    print("cbcr_cov:", cbcr_cov)
    return cbcr_cov


# def face_single_gaussian_model(cbcr, mean, cov):
#     #矩阵行列式 & 逆矩阵
#     covdet = np.linalg.det(cov)
#     covinv = np.linalg.inv(cov)

#     probabilitys = []

#     #---生成数组[[cb,cr], [cb,cr] ... [cb,cr] ]
#     # t1 = time.time()
#     cbcr = np.column_stack((cbcr[0], cbcr[1]))
#     for row in range(np.shape(cbcr)[0]):
#         xdiff = cbcr[row] - mean
#         p = np.exp(-0.5 * np.dot(np.dot(xdiff, covinv), xdiff.T)) / (2 * np.pi * np.power(covdet, 0.5))
#         probabilitys.append(p)

#     # t3 = time.time()

#     # print("single_gaussian_model t1:%f  t2:%f"%(t2-t1,t3-t2))
#     test_show_gaussion_probability(pdfs)


def face_single_gaussian_model(cbcr, mean, cov):
    # t1 = time.time()
    #---生成数组[[cb,cr], [cb,cr] ... [cb,cr] ]
    cbcr = np.column_stack((cbcr[0], cbcr[1]))

    pdfs = multivariate_normal.pdf(cbcr, mean = mean, cov=cov, allow_singular=True)
    # test_show_gaussion_histogram(pdfs, mean, cov)
    global possibly
    possibly = find_gaussion_probability_threshold(pdfs)
 
    # t2 = time.time()
    # print("face_single_gaussian_model cost time:%f"%(t2-t1)) 

def single_gaussian_model(image, position, mean, cov):
    
    #转换为YCbCr 且生成二维数组[[cb,cb,cb,cb,....], [cr,cr,cr,cr,cr,...]]
    cbcr = image_reshape(image)
    cbcr = np.column_stack((cbcr[0], cbcr[1]))

    #计算高斯概率密度分布
    pdfs = multivariate_normal.pdf(cbcr, mean = mean, cov=cov, allow_singular=True)
    # test_show_guaussion_3d(image, pdfs.reshape(image.shape[0], image.shape[1]))

    pdfs[pdfs >= possibly] = 255    #矩阵元素大于阈值设置为白色
    pdfs[pdfs < possibly] = 0       #矩阵小于阈值设置为黑色

    img_gray = pdfs.reshape(image.shape[0], image.shape[1])
    # cv2.imshow("gray",img_gray)
    # cv2.waitKey(0) & 0xFF == ord('q')

    img_gray = close_operation(img_gray)

    
    # cv2.imshow("gray",img_gray)
    # cv2.waitKey(0) & 0xFF == ord('q')
    return  img_gray  


#计算特征点相对位置，因为需截取人脸位置，所以计算特征点相对人脸坐标的位置
def landmark_relative(bbox,landmark):
    for i in range(len(landmark)/2):
        landmark[2*i] -= bbox[0] - 5 # +5是修复检测出的特征点有点向左边偏移的Bug
        landmark[2*i+1] -= bbox[1]
    return landmark

#判断当前像素是否为特征点像素
def is_landmark_pixel(landmark, row, col):
    eye_length = 22.0 #眼睛矩形边长
    eye_width = 32.0
    nose_width = 20.0
    nose_height = 14.0
    mouse_lenght = 20.0

    for i in range(len(landmark)/2):
        if i == 0 or i == 1: #眼睛
            if row >= (landmark[2*i] - eye_width / 2)  and row <= (landmark[2*i] + eye_width / 2) and col >= (landmark[2*i+1] - eye_length / 2) and col <= (landmark[2*i+1] + eye_length / 2):
                return True
        elif i == 2: #鼻子
            if row >= (landmark[2*i] - nose_width / 2) and row <= (landmark[2*i] + nose_width / 2) and col >= (landmark[2*i+1] - nose_height) and col <= (landmark[2*i+1] + 2):
                return True
        else: #左嘴角
            if row >= (landmark[2*i]) and row <= (landmark[2*(i+1)]) and col >= (landmark[2*i+1]) and col >= (landmark[2*(i+1)+1] + mouse_lenght):
                return True
            break

    return False


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

#去除噪音包括5个特征点，以及矩形内接圆之外的区域
def remove_noise(image, landmark, cbcr):
    #中心点坐标
    t1 = time.time()
    circle = (int(image.shape[0]/2), int(image.shape[1]/2))
    radius = image.shape[0]/2

    delete_array = []
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if is_outside_of_circle((row, col), circle, radius) or is_landmark((row, col), landmark):
                # cbcr.pop(row * image.shape[0] + col)
                # np.delete(cbcr,0, row * image.shape[0] + col)
                delete_array.append(row * image.shape[0] + col)
    #删除
    print("delete number,", len(delete_array))
    cbcr = np.delete(cbcr, delete_array, axis=1)

    print("cost time:", time.time() - t1)
    return cbcr

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

def is_outside_of_circle(point, circle, radius):
    (x, y) = point
    (rx,ry) = circle

    distance = np.sqrt(np.square(x - rx) + np.square(y - ry)) 
    return distance > radius

def is_landmark(point, landmark):
    eye_width = 30
    eye_height = 30
    nose_height = 10
    nose_width = 24
    mouse_height = 16

    (x, y) = point
    for i in range(len(landmark)/2):
        if i == 0 or i == 1: #眼睛
            if x >= int(landmark[2*i] - eye_width / 2) and x <= int(landmark[2*i] + eye_width / 2) and y >= int(landmark[2*i+1] - eye_height / 2) and y <= int(landmark[2*i+1] + eye_width / 2): 
                return True
        elif i == 2: #鼻子
            if x >= int(landmark[2*i] - nose_width / 2) and x <= int(landmark[2*i] + nose_width / 2) and y >= int(landmark[2*i+1] - nose_height) and y <= int(landmark[2*i+1] + 2):
                return True 
        else:
            if landmark[2*i+1] <= landmark[2*(i+1)+1]:
                if x >= int(landmark[2*i]) and x <= int(landmark[2*(i+1)]) and y >= int(landmark[2*i+1]) and y <= int(landmark[2*(i+1)+1] + mouse_height): 
                    return True
            else:
                if x >= int(landmark[2*i]) and x <= int(landmark[2*(i+1)]) and y >= int(landmark[2*(i+1)+1]) and y <= int(landmark[2*i+1] + mouse_height): 
                    return True
            break 
    return False

#Gray world 预处理
# def grey_world(image):
#     return  cca.grey_world(image)

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
    
def crop_image(image, bbox):
    
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    left_point_x = int(round(bbox[0] - width)) if int(round(bbox[0] - width)) > 0 else 0
    right_point_x = int(round(bbox[2] + width)) if int(round(bbox[2] + width)) < image.shape[1] else image.shape[1]

    left_point_y = int(round(bbox[1] - height * 0.5)) if int(round(bbox[1] - height * 0.5)) > 0 else 0
    right_point_y = int(round(bbox[3] + height * 1.2 )) if int(round(bbox[3] + height * 1.2)) < image.shape[0] else image.shape[0]

    crop = image[left_point_y:right_point_y, left_point_x : right_point_x]

    Hight = int((250.0 * crop.shape[0]) / crop.shape[1])

    img = cv2.resize(crop, (250, Hight), interpolation=cv2.INTER_CUBIC)
    
    return img

#Test......
def test_svm_classification():
    paths = ["test_positive", "test_negative"]
    n = 100
    meanVal = joblib.load('./features/PCA/%s/meanVal_train_%s.mean' %(m,m)) 
    n_eigVects = joblib.load('./features/PCA/%s/n_eigVects_train_%s_%s.eig' %(m,m,n))  
    model_path = './models/%s/svm_%s_pca_%s.model' %(m,m,n)

    # MTCNN分类器
    mtcnn_detector = mtcnn_detector_init() 

    # SVM 分类器
    clf = joblib.load(model_path)

    num = 0
    total = 0
    t0 = time.time()
    for path in paths:
        lable = 1 if cmp(path, "test_positive") == 0 else 0
        for item in os.listdir(path):
            if item[0] == '.':
                continue

            imagepath = os.path.join(path,item)
            test_data = TestLoader([imagepath])

            all_boxes,landmarks = mtcnn_detector.detect_face(test_data)
            if len(all_boxes[0]) == 0:
                continue

            print("%d Dealing with %s"%(total,imagepath))
            image = cv2.imread(imagepath)

            #Skin Detection
            gray = skinDetect(image, all_boxes[0][0], landmarks[0][0])

            # HOG 
            fd = hog_feature(gray)
            
            feature = fd.reshape((1, -1))
            feature = feature - meanVal
            feature = feature * n_eigVects
            
            result = clf.predict(feature) 
            if int(result) == int(lable):
                num += 1
            total += 1
            # cv2.imshow("lala",image)
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     continue
    rate = float(num) / total
    t1 = time.time()
    print 'The classification accuracy is %f' %rate 
    print("Total time cost is %f"%(t1 - t0))

def test_show_face_area(image, all_boxes):
    for bbox in all_boxes[0]:
        # skinDetect(image, bbox)
        cv2.putText(image,'face',(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
        cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255), 4)


def test_show_landmark_area(image, landmarks):
    for landmark in landmarks[0]:
        for i in range(len(landmark)/2):
            # print("Landmark:%d    %d" ,landmark[2*i] + 5 , landmark[2*i+1])
            cv2.circle(image, (int(landmark[2*i] + 5),int(landmark[2*i+1] + 2)), 3, (0,0,255))

def test_show_ear_area(image, all_boxes):
    for bbox in all_boxes[0]:
        ear_bboxs = cal_ear_area_pt(bbox)
        for j in ear_bboxs:
            corpbbox = [int(j[0]), int(j[1]), int(j[2]), int(j[3])]
            cv2.rectangle(image, (corpbbox[0], corpbbox[1]),
                    (corpbbox[2], corpbbox[3]), (0, 0, 255), 4)

def test_show_landmark_point(image, landmark):
    for i in range(len(landmark)/2):
        print("landmark point:",i)
        cv2.circle(image, (int(landmark[2*i]),int(int(landmark[2*i+1]))), 3, (0,0,255))

    cv2.imshow("landmarl", image)
    cv2.waitKey(0) & 0xFF == ord('q')

def test_show_landmark_modify_area(image, landmark):
    eye_width = 30
    eye_height = 30
    nose_height = 10
    nose_width = 24
    mouse_height = 16

    for i in range(len(landmark)/2):
        if i == 0 or i == 1: #眼睛
            print("eyes")
            cv2.rectangle(image, (int(landmark[2*i] - eye_width / 2),int(landmark[2*i+1] - eye_height / 2)),(int(landmark[2*i] + eye_width / 2),int(landmark[2*i+1] + eye_height/2)),(0,0,255))
        elif i == 2: #鼻子
            print("nose")
            cv2.rectangle(image, (int(landmark[2*i] - nose_width / 2),int(landmark[2*i+1] - nose_height)),(int(landmark[2*i] + nose_width / 2),int(landmark[2*i+1] + 8)),(0,0,255))
        else:
            print("mouse")
            if landmark[2*i+1] <= landmark[2*(i+1)+1]:
                cv2.rectangle(image, (int(landmark[2*i]),int(landmark[2*i+1])),(int(landmark[2*(i+1)]),int(landmark[2*(i+1)+1] + mouse_height)),(0,0,255))
            else:
                cv2.rectangle(image, (int(landmark[2*i]),int(landmark[2*(i+1)+1])),(int(landmark[2*(i+1)]),int(landmark[2*i+1] + mouse_height)),(0,0,255))
            break
    cv2.imshow("landmarl", image)
    cv2.waitKey(0) & 0xFF == ord('q')

# 查找数组中大于特定百分比的数值
# def find_gaussion_probability_threshold(probabilitys):
#     t1 = time.time()
#     n, bins, _ = plt.hist(probabilitys, bins=30)
    
#     # global possibly
#     # possibly = bins[1]
#     # print("possibly:", possibly)
#     sumn = np.sum(n)
#     count = 0
#     for i in range(len(n)):
#         pro = float((n[i]+count)/sumn)
#         print("%.4f:%d:%.2f"%(bins[i], n[i], pro))
#         if pro >= 0.3 and pro <= 0.35:
#             global possibly
#             possibly = bins[i] 
#             print("possibly:", possibly)
#             break
#         count += n[i]
#     print("gaussion_probability cost time:", time.time() - t1)
#     # plt.show()

# 查找数组中大于特定百分比的数值
def find_gaussion_probability_threshold(pdfs):
    t1 = time.time()
    pdfs.sort()
    length = len(pdfs)
    right = len(pdfs) - 1
    left = 0
    while left <= right:
        mid = (right + left) / 2
        rate = float(len(pdfs[pdfs >= pdfs[mid]])) / length
        if rate >= 0.7 and rate <= 0.71:
            break
        else:
            if rate < 0.7:
                right = mid -1
            else:
                left = mid + 1
    t2 = time.time()
    print("find_gaussion_probability_threshold cost time:%f"%(t2-t1))        
    return pdfs[mid]

#测试显示需要去除的特征点位置
def test_show_remove_landmark_piexl(image, landmark):
    eye_length = 22.0 #眼睛矩形边长
    eye_width = 32.0
    nose_width = 20.0
    nose_height = 14.0
    mouse_lenght = 20.0
    for i in range(len(landmark)/2):
        if i == 0 or i == 1:
            cv2.rectangle(image, (int(landmark[2*i] - eye_width / 2),int(landmark[2*i+1] - eye_length / 2)),(int(landmark[2*i] + eye_width / 2),int(landmark[2*i+1] + eye_length / 2)),(0,0,255))
        elif i == 2:
            cv2.rectangle(image, (int(landmark[2*i] - nose_width / 2),int(landmark[2*i+1] - nose_height)),(int(landmark[2*i] + nose_width / 2),int(landmark[2*i+1] + 12)),(0,0,255))
        else:
            cv2.rectangle(image, (int(landmark[2*i]),int(landmark[2*i+1])),(int(landmark[2*(i+1)]),int(landmark[2*(i+1)+1] + mouse_lenght)),(0,0,255))
            break


# 测试显示肤色散列点
def test_show_skin_piexl(x1, y1, color1):
 
    fig = plt.figure(figsize=(8,5))
    plt.scatter(x1, y1, c = color1)

    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.show()

def test_show_guaussion_3d(image,z):
    import mpl_toolkits.mplot3d

    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    _, x, y = cv2.split(img_ycrcb)

    ax = plt.subplot(111, projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow', alpha=0.9)#绘面

    # cset = ax.contour(x,y,z,10,zdir='z',offset = 125,cmap='coolwarm')
    # cset = ax.contour(x, y, z, zdir='x', offset = 0,cmap='winter')
    # cset = ax.contour(x, y, z, zdir='y', offset =255,cmap= 'winter')

    ax.set_xlabel('cb')
    ax.set_ylabel('cr')
    ax.set_zlabel('Probability')
    plt.show()

def test_show_gaussion_histogram(pdfs, mean, cov, num_bins = 50):
    n, bins, patches = plt.hist(pdfs, num_bins, normed=True, facecolor = 'b', alpha = 0.9)

    plt.xlabel('Probability')
    plt.ylabel('Numbers')
    np.set_printoptions(precision = 2)
    title = 'histogram: mean={}'.format(mean),'cov=[{}'.format(cov[0]),'{}]'.format(cov[1])
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    main()