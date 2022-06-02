import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import utils

def train_extract_feat(dir, l_class):
    #apply threshold
    img_binary = utils.set_threshold(dir)
    #give pixel in each component an integer label
    img_label = label(img_binary, background=0)

    regions = regionprops(img_label)
    #collect data
    features = np.array([[0]*utils.num_features])
    labels = []
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        #corner coords

        #denoising: component w less than 10 px
        if (maxc - minc) < 10 or (maxr - minr) < 10:
            continue

        roi = img_binary[minr:maxr, minc:maxc]
        curr_features = utils.extract_feat(roi, props)
        features = np.append(features,curr_features,axis=0)
        labels.append(l_class)

    features = np.delete(features, 0, 0)
    return features, labels;

def training_lab(dir, train_ypred_lab, index_curr):
    img_binary = utils.set_threshold(dir)

    io.imshow(img_binary)

    img_label = label(img_binary, background=0)
    regions = regionprops(img_label)
    ax = plt.gca()

    invert_img_dict = {0:'a', 1:'d', 2:'m', 3:'n', 4:'o', 5:'p', 6:'q', 7:'r', 8:'U', 9:'w'}

    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
        if (maxc - minc) < 10 or (maxr - minr) < 10:
            continue
        center_x = (minr + maxr)/2
        center_y = maxc + 5
        ax.text(center_y, center_x, str(invert_img_dict[train_ypred_lab[index_curr]]), fontsize=8, color='yellow')
        index_curr = index_curr + 1

    io.show()
    return index_curr


def train_img(imgs):
    teach_feat = np.array([[0]*utils.num_features]);
    teach_lab = [];

    for num, images in enumerate(imgs):
        print("Learning Image : ", images)
        features, labels = train_extract_feat("images/" + images,num)
        teach_feat = np.append(teach_feat,features,axis=0)
        teach_lab.extend(labels)

    teach_feat = np.delete(teach_feat, 0, 0)

    mean = np.mean(teach_feat, axis = 0)
    std = np.std(teach_feat, axis = 0)

    teach_feat = (teach_feat - mean)/std


    #find distance between each character and all other characters
    D = cdist(teach_feat, teach_feat)
    io.imshow(D)
    plt.title('Distance Matrix')
    io.show()
    D_index = np.argsort(D, axis=1)

    train_ypred_lab = utils.classify(D_index, teach_lab)

    #Train given images
    correct = 0
    for i in range(0,len(teach_lab)):
        if(teach_lab[i] == train_ypred_lab[i]):
            correct = correct+1
    print("training recognition rate: ", correct/len(teach_lab), len(teach_lab))

    # Confusion matrix
    confM = confusion_matrix(teach_lab, train_ypred_lab)
    io.imshow(confM)
    plt.title('Confusion Matrix')
    io.show()
#    Show training images

    index_curr = 0
    for num, images in enumerate(imgs):
        index_curr = training_lab("images/" + images, train_ypred_lab, index_curr)

    return teach_feat, teach_lab, mean, std;
