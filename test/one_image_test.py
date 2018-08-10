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
        for bbox in all_boxes[count]:
            skinDetect(image, bbox)
            cv2.putText(image,str(np.round(bbox[4],2)),(int(bbox[0]),int(bbox[1])),cv2.FONT_HERSHEY_TRIPLEX,1,color=(255,0,255))
            cv2.rectangle(image, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255))
            
        for landmark in landmarks[count]:
            for i in range(len(landmark)/2):
                cv2.circle(image, (int(landmark[2*i]),int(int(landmark[2*i+1]))), 3, (0,0,255))
            
        count = count + 1
        # cv2.imwrite("%d.png" %(count),image)
        cv2.imshow("lala",image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            continue   

##################################################
#自适应肤色检测，根据面部肤色区域，计算均值 & 协方差矩阵
#高斯模型概率密度分布
##################################################
def skinDetect(image, bbox):
    #裁剪出脸部区域
    crop_image = image[int(bbox[1]): int(bbox[3]), int(bbox[0]): int(bbox[2])]
    
    img_ycrcb = cv2.cvtColor(crop_image, cv2.COLOR_BGR2YCrCb)
    
    cbcr_mean = mean(img_ycrcb)

    cbcr_cov = covariance_matrix(img_ycrcb)
    
    single_gaussian_model(image, cbcr_mean, cbcr_cov)

##################################################
#计算均值
##################################################
def mean(img_ycrcb):
    y, cr, cb = cv2.split(img_ycrcb)
 
    cr_mean = np.mean(cr)
    cb_mean = np.mean(cb)

    cbcr_mean = np.mat([cb_mean, cr_mean])
    # print("cbcr_mean:", cbcr_mean)


    return cbcr_mean

def covariance_matrix(img_ycrcb):
    y, cr, cb = cv2.split(img_ycrcb)

    crravel = cr.ravel()
    cbravel = cb.ravel()

    #---生成数组[[cb,cr], [cb,cr] ... [cb,cr] ]
    # cbcr = np.column_stack((cbravel, crravel)) 
    # cov_sum = np.mat([0, 0])
    # for row in range(np.shape(cbcr)[0]):
    #     xdiff = cbcr[row] - cbcr_mean
    #     xdiff_mat = np.mat(xdiff)
    #     cov_sum = cov_sum + np.dot(xdiff_mat.T, xdiff_mat)

    # print("cov_sum:", cov_sum)
    # cov = cov_sum / (np.shape(cbcr)[0] - 1)
    # print("cov:",cov)
    cbcr = np.vstack((crravel, cbravel))
    # print("cbcr:", cbcr)

    cbcr_cov = np.cov(cbcr)
    # print("cbcr_cov:", cbcr_cov)

    return np.mat(cbcr_cov)


def single_gaussian_model(image, mean, cov):
    img_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print("img_gray:", img_gray)
   

    #Test......
    # mean = np.mat([124.2125, 132.9449])
    # cov = np.mat([[75.3881,40.2587], [40.2587, 250.2942]])

    y, cr, cb = cv2.split(img_ycrcb)

    crravel = cr.ravel()
    cbravel = cb.ravel()
    imgravel = img_gray.ravel()

    #矩阵行列式 & 逆矩阵
    covdet = np.linalg.det(cov)
    covinv = np.linalg.inv(cov)

    #---生成数组[[cb,cr], [cb,cr] ... [cb,cr] ]
    cbcr = np.column_stack((cbravel, crravel)) 
    for row in range(np.shape(cbcr)[0]):
        xdiff = cbcr[row] - mean
        p = np.exp(-0.5 * np.dot(np.dot(xdiff, covinv), xdiff.T)) / (2 * np.pi * np.power(covdet, 0.5))
        if p >= 0.0015:
            imgravel[row] = 255
        else:
            imgravel[row] = 0

    img_gray = imgravel.reshape(np.shape(img_gray)[0], np.shape(img_gray)[1])

    cv2.imshow("gray",img_gray)
    cv2.waitKey(0) & 0xFF == ord('q')



if __name__ == "__main__":
    main()