from skimage.filters import threshold_otsu, threshold_li, threshold_yen
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
import cv2
import numpy as np
from skimage import io
from collections import Counter

num_features = 12
knn_neighbors = 3

def extract_feat(roi, props):
    features = []
    m = moments(roi)

    # Computing	Hu	Moments and	Removing	Small	Components

    cr = m[0, 1] / m[0, 0]
    cc = m[1, 0] / m[0, 0]
    mu = moments_central(roi, center=(cr, cc))
    nu = moments_normalized(mu)
    hu = moments_hu(nu)

    features.extend(hu)
    features.append(roi.shape[1]/roi.shape[0])
    features.append(props.eccentricity)
    features.append(props.convex_area/props.area)
    features.append(props.orientation)
    features.append(props.euler_number)

    return np.array([features])


def classify(D_index, train_labels):
#    k_nearest_neigbors
    pred_labels_test = []
    for idx,row in enumerate(D_index):
        k_labels = [train_labels[row[i]] for i in range(0,knn_neighbors)]
        most_common,num_most_common = Counter(k_labels).most_common(1)[0]
        pred_labels_test.append(most_common)
    return pred_labels_test

def set_threshold(dir):
    img = cv2.imread(dir,0)
    blur = cv2.GaussianBlur(img,(5,5),0)

#    Manual Thresholding
#    th = 150

#    Otsu Thresholding
#    th = threshold_otsu(img)

#    img_binary = (img < th).astype(np.double)

#    Adaptive Thresholding

    img_binary = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
#
#    th,img_binary = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    th,img_binary = cv2.threshold(blur,200,255,cv2.THRESH_BINARY)

    kernel = np.ones((3,3),np.uint8)
    #3x3 kernel
    closing = cv2.morphologyEx(~img_binary, cv2.MORPH_CLOSE, kernel)

    return closing
