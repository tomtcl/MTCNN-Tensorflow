#coding:utf-8
import sys
import time
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
<<<<<<< HEAD
import time
import matplotlib.pyplot as plt
=======
import matplotlib.pyplot as plt
import colorcorrect.algorithm as cca

>>>>>>> remove pyc files

np.set_printoptions(threshold='nan') 

def main():
    
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
    gt_imdb = []
    #gt_imdb.append("35_Basketball_Basketball_35_515.jpg")
    #imdb_ = dict()"
    #imdb_['image'] = im_path
    #imdb_['label'] = 5
    path = "tel"
    for item in os.listdir(path):
        if item[0] != '.':
            gt_imdb.append(os.path.join(path,item))
    test_data = TestLoader(gt_imdb)
    all_boxes,landmarks = mtcnn_detector.detect_face(test_data)
    count = 0
    # imagepath = gt_imdb[0]
    for imagepath in gt_imdb:
        print(imagepath)
        image = cv2.imread(imagepath)
        # for bbox in all_boxes[count]:
        #     # skinDetect(image, bbox)
        #     cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
        #     cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
            
        # for landmark in landmarks[count]:
        #     for i in range(len(landmark)/2):
        #         # print("Landmark:%d    %d" ,landmark[2*i] + 5 , landmark[2*i+1])
        #         cv2.circle(image, (int(landmark[2*i] + 5),int(int(landmark[2*i+1]))), 3, (0,0,255))
        skinDetect(image, all_boxes[count][0], landmarks[count][0])

        count = count + 1
        # cv2.imwrite("%d.png" %(count),image)
        # cv2.imshow("lala",image)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
            # continue   

##################################################
#自适应肤色检测，根据面部肤色区域，计算均值 & 协方差矩阵
#高斯模型概率密度分布
##################################################
def skinDetect(image, bbox, landmark):
    #裁剪出脸部区域
    crop_image = image[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])]
    
    landmark = relative_position_landmark(bbox, landmark)

    # test_show_landmark_area(crop_image, landmark)

    img_ycrcb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2YCrCb)
    cbcr = resize_image_shape(img_ycrcb)

    #去除噪声
    cbcr = remove_noise(crop_image, landmark, cbcr)

    cbcr_mean = mean(cbcr)

    cbcr_cov = covariance_matrix(cbcr)
    # TEST------
    face_single_gaussian_model(cbcr, cbcr_mean, cbcr_cov)

    #计算人脸区域协方差
    cbcr_cov = covariance_matrix(img_ycrcb, cbcr_mean, landmark)
    
    #计算整张图片的单峰高斯概率密度pdf
    single_gaussian_model(crop_image, cbcr_mean, cbcr_cov)

##################################################
#采用mean公式计算均值
##################################################
def mean(cbcr):
    # y, cr, cb = cv2.split(img_ycrcb)
 
    # cr_mean = np.mean(cr)
    # cb_mean = np.mean(cb)

    # cbcr_mean = [cb_mean, cr_mean]
    cbcr_mean = np.mean(cbcr, axis=1)
    print("cbcr_mean:", cbcr_mean)

    return cbcr_mean

def covariance_matrix(cbcr):
    # y, cr, cb = cv2.split(img_ycrcb)

    # crravel = cr.ravel()
    # cbravel = cb.ravel()

    # cbcr = np.vstack((crravel, cbravel))
    # print("cbcr:", cbcr)

    cbcr_cov = np.cov(cbcr)
    print("cbcr_cov:", cbcr_cov)

    return cbcr_cov
##################################################
#自定义计算均值
##################################################
# def mean(img_ycrcb, landmark):
#     cost = time.time()    
#     cov_sum = np.mat([0.0, 0.0])
#     landmark_pixel_sum = 0
#     for row in range(np.shape(img_ycrcb)[0]):
#         for col in range(np.shape(img_ycrcb)[1]):
#             _, cr, cb = img_ycrcb[row][col]
#             if is_landmark_pixel(landmark, row, col):
#                 landmark_pixel_sum += 1
#                 continue
#             cov_sum = cov_sum + np.mat([cb, cr])
#     cbcr_mean = cov_sum / ((np.shape(img_ycrcb)[0] * np.shape(img_ycrcb)[1]) - landmark_pixel_sum)
#     print("cbcr_mean:", cbcr_mean)
#     cost = time.time() - cost
#     print("mean cost time:", cost)
#     return cbcr_mean

##################################################
# 采用numpy自带公式计算协方差矩阵
##################################################
def covariance_matrix(img_ycrcb,cbcr_mean,landmark):
    cost = time.time()
    _, cr, cb = cv2.split(img_ycrcb)
    
    # 转换成一维数组
    crravel = cr.ravel()
    cbravel = cb.ravel()

    #---生成数组[[cb,cr], [cb,cr] ... [cb,cr] ]
    cbcr = np.vstack((crravel, cbravel))
    # print("cbcr:", cbcr)

    cbcr_cov = np.cov(cbcr)
    print("cbcr_cov:", cbcr_cov)

    cost = time.time() - cost
    print("covariance_matrix cost time:", cost)

    return cbcr_cov

##################################################
# 自定义协方差计算
##################################################
# def covariance_matrix(img_ycrcb, cbcr_mean,landmark):
#     cost = time.time()
#     cov_sum = np.mat([0.0, 0.0])
#     landmark_pixel_sum = 0
#     for row in range(np.shape(img_ycrcb)[0]):
#         for col in range(np.shape(img_ycrcb)[1]):
#             if is_landmark_pixel(landmark, row, col):
#                 landmark_pixel_sum += 1
#                 continue
#             _, cr, cb = img_ycrcb[row][col]
#             xdiff = np.mat([cb, cr]) - cbcr_mean
#             cov_sum = cov_sum + np.dot(xdiff.T, xdiff)
    
#     cbcr_cov = cov_sum / (np.shape(img_ycrcb)[0] * np.shape(img_ycrcb)[1] - landmark_pixel_sum - 1)
#     print("cbcr_cov:",cbcr_cov)

#     cost = time.time() - cost
#     print("covariance_matrix cost time:", cost)

#     return np.mat(cbcr_cov)


def single_gaussian_model(image, mean, cov):
    cost = time.time()

    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print("img_gray:", img_gray)
   
    # #Test......
    # # mean = np.mat([123.651, 51.568])
    # # cov = np.mat([[441.23,5.15], [5.15, 289.12]])

    _, cr, cb = cv2.split(img_ycrcb)

    crravel = cr.ravel()
    cbravel = cb.ravel()
    imgravel = img_gray.ravel()

    # #矩阵行列式 & 逆矩阵
    covdet = np.linalg.det(cov)
    covinv = np.linalg.inv(cov)

    # #---生成数组[[cb,cr], [cb,cr] ... [cb,cr] ]
    cbcr = np.column_stack((cbravel, crravel)) 
    pdfs = []
    for row in range(np.shape(cbcr)[0]):
        xdiff = cbcr[row] - mean
        p = np.exp(-0.5 * np.dot(np.dot(xdiff, covinv), xdiff.T)) / (2 * np.pi * np.power(covdet, 0.5))
        pdfs.append(p)
        if p >= 0.00418:
            imgravel[row] = 255
        else:
            imgravel[row] = 0 #非人脸部分填充黑色
    #Test......
    print("prox:",len(pdfs), "sum:", sum(pdfs))
    test_show_hist(pdfs)

    img_gray = imgravel.reshape(np.shape(img_gray)[0], np.shape(img_gray)[1])

    cost = time.time() - cost
    print("single_gaussian_model cost time:", cost)

    cv2.imshow("gray",img_gray)
    cv2.waitKey(0) & 0xFF == ord('q')

    

    # test_show_skin_piexl(np.array(proxs), np.array(proxs), 'b')
    # img_open = open_operation(img_gray)
    # cv2.imshow("open",img_open)
    # cv2.waitKey(0) & 0xFF == ord('q')

    # img_close = close_operation(img_open)
    # cv2.imshow("close",img_close)
    # cv2.waitKey(0) & 0xFF == ord('q')


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

#Gray world 预处理
def grey_world(image):
    return  cca.grey_world(image)

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


    plt.xlim([60, 150])
    plt.ylim([90, 180])
    plt.show()

<<<<<<< HEAD
def face_single_gaussian_model(cbcr, mean, cov):
    #矩阵行列式 & 逆矩阵
    covdet = np.linalg.det(cov)
    covinv = np.linalg.inv(cov)

    probabilitys = []

    #---生成数组[[cb,cr], [cb,cr] ... [cb,cr] ]
    cbcr = np.column_stack((cbcr[0], cbcr[1]))
    for row in range(np.shape(cbcr)[0]):
        xdiff = cbcr[row] - mean
        p = np.exp(-0.5 * np.dot(np.dot(xdiff, covinv), xdiff.T)) / (2 * np.pi * np.power(covdet, 0.5))
        probabilitys.append(p)

    test_show_gaussion_probability(probabilitys)

# 生成二维数组[[cb,cb,cb,cb,....], [cr,cr,cr,cr,cr,...]]
def resize_image_shape(image):
    _, cr, cb = cv2.split(image)
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

    # #cbcr转换为一维数组[[cb,cr],[cb,cr],...]
    # cbcr = np.column_stack((cbcr[0], cbcr[1])) 
    print("cbcr shape", cbcr.shape)

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

def is_outside_of_circle(point, circle, radius):
    (x, y) = point
    (rx,ry) = circle

    distance = np.sqrt(np.square(x - rx) + np.square(y - ry)) 
    return distance > radius

def relative_position_landmark(bbox, landmark):
    for i in range(len(landmark)/2):
        landmark[2*i] -= bbox[0] - 5
        landmark[2*i+1] -= bbox[1] - 5
    return landmark

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

#Test......
def test_show_landmark_point(image, landmark):
    for i in range(len(landmark)/2):
        print("landmark point:",i)
        cv2.circle(image, (int(landmark[2*i]),int(int(landmark[2*i+1]))), 3, (0,0,255))

    cv2.imshow("landmarl", image)
    cv2.waitKey(0) & 0xFF == ord('q')

def test_show_landmark_area(image, landmark):
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
    
def test_show_gaussion_probability(probabilitys):

    plt.hist(probabilitys, bins=30)
=======
# 测试显示直方图
def test_show_hist(x):
    plt.figure("hist")
    n, bins, patches = plt.hist(x)
>>>>>>> remove pyc files
    plt.show()

if __name__ == "__main__":
    main()