

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
from keras.optimizers import SGD
import imutils


import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)



def identity_block(X, f, filters, stage, block):


    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'


    F1, F2, F3 = filters


    X_shortcut = X

    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size= (f, f), strides= (1, 1), padding ='same', name = conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)


    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)



    return X




def convolutional_block(X, f, filters, stage, block, s=2):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters


    X_shortcut = X


    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)


    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)


    return X


tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = convolutional_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    print("out = " + str(out[0][1][1][0]))




def ResNet50(input_shape=(64, 64, 3), classes=2):

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')


    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    X = X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)


    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model



model = ResNet50(input_shape = (75, 75, 3), classes = 2)




model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])




import pandas as pd
from keras.utils import*



def load_dataset(path='/Users/dhruvsamant/Desktop/MLProject/Resnet/train.json'):
     train=pd.read_json(path)
     train_images = train.apply(lambda c_row: [np.stack([c_row['band_1'],c_row['band_2']], -1).reshape((75,75,2))],1)
     train_images = np.stack(train_images).squeeze()
     return train, train_images


from scipy.ndimage.filters import uniform_filter
# despecle the noise

def decibel_to_linear(band):
     # convert to linear units
    return np.power(10,np.array(band)/10)

def linear_to_decibel(band):
    return 10*np.log10(band)



def lee_filter(band, window, var_noise = 0.25):
    mean_window = uniform_filter(band, window)
    mean_sqr_window = uniform_filter(band**2, window)
    var_window = mean_sqr_window - mean_window**2

    weights = var_window / (var_window + var_noise)
    band_filtered = mean_window + weights*(band - mean_window)
    return band_filtered


def create_set(train_images,train):
    m,n,o,p = np.shape(train_images)
    images=[]
    y= np.zeros((m,1))
    for i in range(m):
        b1=train_images[i,:,:,0]
        b2=train_images[i,:,:,1]
        b3=b1/b2
        r=(b1-b1.min())/(b1.max()-b1.min())
        #r = r[5:69,5:69]
        g=(b2-b2.min())/(b2.max()-b2.min())
        #g = g[5:69,5:69]
        b=(b3-b3.min())/(b3.max()-b3.min())
        #b = b[5:69,5:69]
        y[i] = train[i]
        final = np.dstack((r, g, b))
        images.append(final)
    images=np.asarray(images)
    return images,y


# rotate only the inscribed circle of the image
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

# flip the image horizontally
def mirror(image):
    copy = np.copy(image)
    m = len(copy)
    n = len(copy[0])
    for i in range(m):
        for j in range(n//2):
            copy[i][j], copy[i][n-j-1] = copy[i][n-j-1], copy[i][j]
    return copy

def shift(image, width_rg=15, height_rg=15, u=0.5, v=1.0):

    if v < u:
        image = prep.random_shift(image, wrg=width_rg, hrg=height_rg,
                                  row_axis=0, col_axis=1, channel_axis=2)

    return image

def augment(imset, aug_size):
    # first augment by rotation
    n = len(imset)
    copy = np.copy(imset)
    augmented = np.zeros((n*(aug_size-1),75,75,2))
    angel = 360//aug_size
    m = 0
    print("Rotating images")
    for i in range(n):
        e = imset[i,:,:,:]
        for j in range(aug_size-1):
            m += 1
            if m%2000 == 0:
                print('{percent:.2%}'.format(percent=m/((aug_size-1)*n)))
            augmented[j*n+i,:,:,0] = rotate(e[:,:,0], angel*(j+1))
            augmented[j*n+i,:,:,1] = rotate(e[:,:,1], angel*(j+1))
    augmented = np.concatenate((copy, augmented))

    # then augment by mirroring
    mirrored = np.zeros((n*aug_size,75,75,2))
    n2 = n*aug_size
    m = 0
    print("\nFlipping images")
    for i in range(n2):
        e = augmented[i,:,:,:]
        m += 1
        if m%2000 == 0:
            print('{percent:.2%}'.format(percent=m/n2))
        mirrored[i,:,:,0] = mirror(e[:,:,0])
        mirrored[i,:,:,1] = mirror(e[:,:,1])
    augmented = np.concatenate((augmented, mirrored))
    print(augmented.shape)
    return augmented

Ytest= np.zeros(len(Y_test))

def augment_y(yset, n):
    augmentedy = np.zeros(len(yset)*n)
    for i in range(len(augmentedy)):
        augmentedy[i] = yset[i%(len(yset))]
    print("augmented y", augmentedy.shape)
    return augmentedy

def decibel_to_linear(band):
     # convert to linear units
    return np.power(10,np.array(band)/10)

def linear_to_decibel(band):
    return 10*np.log10(band)


    # implement the Lee Filter for a band in an image already reshaped into the proper dimensions
def lee_filter(band, window, var_noise = 0.25):
    # band: SAR data to be despeckled (already reshaped into image dimensions)
    # window: descpeckling filter window (tuple)
    # default noise variance = 0.25
    # assumes noise mean = 0


    weights = var_window / (var_window + var_noise)
    band_filtered = mean_window + weights*(band - mean_window)
    return band_filtered



train, train_images = load_dataset()


augmented = augment(train_images, 12)
augmented_y = augment_y(train.iloc[:,4], 24)



X_train, Y_train = create_set(augmented,augmented_y)
Y_train = to_categorical(Y_train,num_classes=2)


X_test, Y_test = create_set(Xv,yv.iloc[:,4])



from sklearn.model_selection import train_test_split


Xtr, Xv, ytr, yv = train_test_split(X_train, Y_train, test_size= 0.1)



history=model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs = 10, batch_size=32)


test, test_images = load_dataset('/Users/dhruvsamant/Desktop/MLProject/Resnet/test.json')
X_test, Y_test = create_set(test_images,test)
Y_test = to_categorical(Y_test[:,4],num_classes=2)





import matplotlib.pyplot as plt
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
pri




model.summary()



plot_model(model, to_file='model.png')
SVG(model_to_dot(model).create(prog='dot', format='svg'))


for i in range(len(y)):
    if y[i,0] >= y[i,1]:
        y[i,0] = 1
        y[i,1] = 0
    else:
        y[i,0] = 0
        y[i,1] = 1
