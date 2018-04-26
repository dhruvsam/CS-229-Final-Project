import os
import numpy as np
import pandas as pd
from skimage.util.montage import montage2d
import matplotlib.pyplot as plt
import imutils
import cv2
import math
import random
import json
base_path = os.path.join('..', 'input')

# concatenate and reshape
def load_and_format(in_path):
    out_df = pd.read_json(in_path)
    out_images = out_df.apply(lambda c_row: [np.stack([c_row['band_1'],c_row['band_2']], -1).reshape((75,75,2))],1)
    out_images = np.stack(out_images).squeeze()
    return out_df, out_images

def rotate(image, angel):
    copy = np.copy(image)
    circled = copy
    edge = []
    r = len(image)/2
    for i in range(len(image)):
        edge.append([])
        for j in range(len(image[0])):
            if math.sqrt((i-r)**2+(j-r)**2) >= r-3:
                edge[i].append(circled[i][j])
            else:
                edge[i].append(0)
            if math.sqrt((i-r)**2+(j-r)**2) > r:
                circled[i][j] = 0
    rotated = imutils.rotate(circled, angel)
    for i in range(len(image)):
        for j in range(len(image[0])):
            if math.sqrt((i-r)**2+(j-r)**2) >= r-3:
                rotated[i][j] = edge[i][j]
    return rotated

def augment(imset):
    #augmented = np.copy(imset)
    n = len(imset)
    copy = np.copy(imset)
    augmented = np.zeros((n*11,75,75,2))
    m = 0
    for i in range(n):
        e = imset[i,:,:,:]
        for j in range(11):
            m += 1
            if m%1000 == 0:
                print('{percent:.2%}'.format(percent=m/(11*n)))
            augmented[j*n+i,:,:,0] = rotate(e[:,:,0], 30*(j+1))
            augmented[j*n+i,:,:,1] = rotate(e[:,:,1], 30*(j+1))
    augmented = np.concatenate((copy,augmented))
    #print(augmented.shape)
    return augmented

def testaugmented(augmented, size):
    n = random.randint(0,size-1)
    m = random.randint(0,1)
    print(n,m)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
    ax1.matshow(augmented[n,:,:,m])
    ax1.set_title('before')
    ax2.matshow(augmented[n+size*2,:,:,m])
    ax2.set_title('60 degrees')

    fig, (ax3, ax4) = plt.subplots(1,2, figsize = (12, 6))
    ax3.matshow(augmented[n+size*4,:,:,m])
    ax3.set_title('120 degress')
    ax4.matshow(augmented[n+size*6,:,:,m])
    ax4.set_title('180 degress')

    fig, (ax5, ax6) = plt.subplots(1,2, figsize = (12, 6))
    ax5.matshow(augmented[n+size*8,:,:,m])
    ax5.set_title('240 degress')
    ax6.matshow(augmented[n+size*10,:,:,m])
    ax6.set_title('300 degrees')

    plt.show()


def main():
    train_df, train_images = load_and_format('/Users/dhruvsamant/Desktop/MLProject/Resnet/train.json')
    print('training', train_df.shape, 'loaded', train_images.shape)
    #test_df, test_images = load_and_format(os.path.join(base_path, 'test.json'))
    #print('testing', test_df.shape, 'loaded', test_images.shape)



    ####### here is the augmented data #######
    augmented = augment(train_images)
    #testaugmented(augmented, len(train_images))
    print("new size",augmented.shape)

    with open('augmented.json', 'w') as f:
        json.dump(augmented.tolist(), f)
    print("done writing")



    #with open('augmented.json', 'w') as f:
    #    json.dump(augmented.tolist(), f)
    #print("done writing")

    #with open('augmented.json', 'r') as g:
    #    data = json.load(g)
    #a = np.array(data)

    #print(a.shape)


main()

##
# loop over the rotation angles again, this time ensuring
# no part of the image is cut off
#for angle in np.arange(0, 360, 15):
#    rotated = imutils.rotate_bound(image, angle)
#    cv2.imshow("Rotated (Correct)", rotated)
#3    cv2.waitKey(0)

### training data overview
##fig, (ax1s, ax2s) = plt.subplots(2,2, figsize = (8,8))
##obj_list = dict(ships = train_df.query('is_iceberg==0').sample(16).index,
##     icebergs = train_df.query('is_iceberg==1').sample(16).index)
##for ax1, ax2, (obj_type, idx_list) in zip(ax1s, ax2s, obj_list.items()):
##    ax1.imshow(montage2d(train_images[idx_list,:,:,0]))
##    ax1.set_title('%s Band 1' % obj_type)
##    ax1.axis('off')
##    ax2.imshow(montage2d(train_images[idx_list,:,:,1]))
##    ax2.set_title('%s Band 2' % obj_type)
##    ax2.axis('off')
##
### testing data overview
##fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12,12))
##idx_list = test_df.sample(49).index
##obj_type = 'Test Data'
##ax1.imshow(montage2d(test_images[idx_list,:,:,0]))
##ax1.set_title('%s Band 1' % obj_type)
##ax1.axis('off')
##ax2.imshow(montage2d(test_images[idx_list,:,:,1]))
##ax2.set_title('%s Band 2' % obj_type)
##ax2.axis('off')

#plt.show()
