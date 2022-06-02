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

    img_dict = {'a':0, 'd':1,'m':2,
                 'n':3, 'o':4, 'p':5,
                 'q':6, 'r':7, 'U':8, 'w':9}

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

    invert_img_dict = {0:'a', 1:'d', 2:'m',
                     3:'n', 4:'o', 5:'p',
                     6:'q', 7:'r', 8:'U', 9:'w'}

    for index,box in enumerate(arr_bb):
        ctrx = (box[0] + box[2])/2
        ctry = box[3] + 10
        ax.text(ctry, ctrx, str(invert_img_dict[pred_labels_test[index]]), fontsize=15, color='white')
    io.show()

    return arr_bb, pred_labels_test
