import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import utils


def test_extract_feat(dir):

    #display binary
    img_binary = utils.set_threshold(dir)

    io.imshow(img_binary)

    #give pixel in each component an integer label
    img_label = label(img_binary, background=0)

    #display component bounding boxes
    print ('Connected Components Found: ', np.amax(img_label))
    regions = regionprops(img_label)
    ax = plt.gca()
    #collect data
    features = np.array([[0]*utils.num_features])
    arr_bb = np.array([[0,0,0,0]])

    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        #corner coords

        #denoising: component w less than 10 px
        if (maxc - minc) < 10 or (maxr - minr) < 10:
            continue

        #bounding box
        ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
        roi = img_binary[minr:maxr, minc:maxc]

        curr_features = utils.extract_feat(roi, props)
        features = np.append(features,curr_features,axis=0)
        arr_bb = np.append(arr_bb, np.array([[minr, minc, maxr, maxc]]),axis=0)

    print ('Identied Components: ', len(features)-1)

    features = np.delete(features, 0, 0)
    arr_bb = np.delete(arr_bb, 0, 0)

    return features,arr_bb;


def eval_recog_rate(gtfile, arr_bb, labels):
    #load pickle
    with open(gtfile, 'rb') as pkl_file:
        mydict = pickle.load(pkl_file,encoding='latin1')
    pkl_file.close()
    classes = mydict['classes']
    locations = mydict['locations']

    img_dict = {'a':0, 'd':1,'m':2,'n':3, 'o':4, 'p':5,'q':6, 'r':7, 'U':8, 'w':9}

    #eval
    correct = 0
    #ground trustlabels for all characters and locations containing ctr coords
    for index,ctr in enumerate(locations):
        for loc,box in enumerate(arr_bb):
            if (ctr[1] > box[0]) & (ctr[1] < box[2]) & (ctr[0] > box[1]) & (ctr[0] < box[3]) & (img_dict[classes[index]] == labels[loc]):
                correct = correct + 1
    print ("Generated recognition rate for given test image: ", correct/len(classes))


def test(image, teach_feat, teach_lab, mean, std):
    features, arr_bb = test_extract_feat("images/" + image)
    features = (features - mean)/std

    D = cdist(features, teach_feat)
    D_index = np.argsort(D, axis=1)

    pred_labels_test = utils.classify(D_index, teach_lab)
    ax = plt.gca()

    invert_img_dict = {0:'a', 1:'d', 2:'m',3:'n', 4:'o', 5:'p', 6:'q', 7:'r', 8:'U', 9:'w'}

    for index,box in enumerate(arr_bb):
        ctrx = (box[0] + box[2])/2
        ctry = box[3] + 10
        ax.text(ctry, ctrx, str(invert_img_dict[pred_labels_test[index]]), fontsize=15, color='white')
    io.show()

    return arr_bb, pred_labels_test

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
        ax.text(center_y, center_x, str(invert_img_dict[train_ypred_lab[index_curr]]), fontsize=10, color='white')
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

# Use training images
imgs = ["a.bmp", "d.bmp", "m.bmp","n.bmp", "o.bmp", "p.bmp","q.bmp", "r.bmp", "u.bmp", "w.bmp"];
teach_feat, teach_lab, mean, std =train_img(imgs)
# Test recognition rate using ground truth
bbox_list, labels = test("test.bmp", teach_feat, teach_lab, mean, std)
eval_recog_rate('test_gt.pkl', bbox_list, labels)
