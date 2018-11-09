import numpy as np
import cv2
import os
import itertools
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, UpSampling2D, Flatten
from keras import backend as K
from keras.optimizers import Adam
from PIL import Image

model = Sequential()
train_images="C:\\Users\\Nexus\\Desktop\\Diplom\\train_full"
train_images_full="C:\\Users\\Nexus\\Desktop\\Diplom\\train_full"

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def load_image(path):
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def feature_matrix_con(img):
    
    return (img / 255.0)

def feature_matrix_full(img):
    
    window = (5, 5)
    nbrs = [cv2.getRectSubPix(img, window, (y, x)).ravel() 
            for x, y in itertools.product(range(np.shape(img)[0]), range(np.shape(img)[1]))]

    median5 = cv2.medianBlur(img, 5).ravel()
    median25 = cv2.medianBlur(img, 25).ravel()
    grad = np.abs(cv2.Sobel(img, cv2.CV_16S, 1, 1, ksize=3).ravel())
    div = np.abs(cv2.Sobel(img, cv2.CV_16S, 2, 2, ksize=3).ravel())
    misc = np.vstack((median5, median25, grad, div)).transpose()

    features = np.hstack((np.asarray(nbrs), misc))
    return (features / 255.0).astype('float32')
    

def target_matrix_con(img):
      
    return (img / 255.0)

def target_matrix_full(img):
       
    return (img / 255.0).astype('float32').ravel()[:, None]

def load_train_set_con(file_list):
    xs = []
    ys = []
    for fname in file_list:
        xs.append(feature_matrix_con(load_image(os.path.join('C:\\Users\\Nexus\\Desktop\\Diplom\\train', fname))))
        ys.append(target_matrix_con(load_image(os.path.join('C:\\Users\\Nexus\\Desktop\\Diplom\\train_cleaned', fname))))
    return np.vstack(xs), np.vstack(ys)


def load_train_set_full(file_list):
    xs = []
    ys = []
    for fname in file_list:
        xs.append(feature_matrix_full(load_image(os.path.join('C:\\Users\\Nexus\\Desktop\\Diplom\\train_full', fname))))
        ys.append(target_matrix_full(load_image(os.path.join('C:\\Users\\Nexus\\Desktop\\Diplom\\train_cleaned', fname))))
    return np.vstack(xs), np.vstack(ys)

def blockshaped(arr, nrows, ncols): 
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
    

def unet(img_channels, image_rows, image_cols):
    inputs = Input((img_channels, image_rows, image_cols))
    conv1 = Conv2D(32, (3, 3), activation='tanh', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='tanh', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='tanh', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='tanh', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='tanh', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='tanh', padding='same')(conv3)
    
    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='tanh', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='tanh', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='tanh', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='tanh', padding='same')(conv9)
    

    
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

    return model


def train_con():
       
    model = unet(112,112,1)
    
    files = os.listdir(train_images) 
    for i in files:
        print("0) ",i)
        train_x, train_y = load_train_set_con([i]) 
        train_x = cv2.resize(train_x, (448, 448)) 
        train_y = cv2.resize(train_y, (448, 448))
        train_x = blockshaped(train_x,112,112)
        train_y = blockshaped(train_y,112,112)
        train_x = np.expand_dims(train_x, axis=3)
        train_y = np.expand_dims(train_y, axis=3)
        model.fit(train_x, train_y, batch_size=16,epochs=20, validation_split=0.1, verbose=0)
     
    model.save('weights_1.h5')                                                    
    files = os.listdir(train_images) 
    for i in files:
        print("90) ",i)
        train_x, train_y = load_train_set_con([i]) 
        train_x = np.rot90(train_x, 3)
        train_y = np.rot90(train_y, 3)
        train_x = cv2.resize(train_x, (448, 448)) 
        train_y = cv2.resize(train_y, (448, 448))
        train_x = blockshaped(train_x,112,112)
        train_y = blockshaped(train_y,112,112)
        train_x = np.expand_dims(train_x, axis=3)
        train_y = np.expand_dims(train_y, axis=3)
        model.fit(train_x, train_y, batch_size=16,epochs=20, validation_split=0.1, verbose=0)
    
    model.save('weights_2.h5') 
    files = os.listdir(train_images) 
    for i in files:
        print("180) ",i)
        train_x, train_y = load_train_set_con([i]) 
        train_x = np.rot90(train_x, 3)
        train_x = np.rot90(train_x, 3)
        train_y = np.rot90(train_y, 3)
        train_y = np.rot90(train_y, 3)
        train_x = cv2.resize(train_x, (448, 448)) 
        train_y = cv2.resize(train_y, (448, 448))
        train_x = blockshaped(train_x,112,112)
        train_y = blockshaped(train_y,112,112)
        train_x = np.expand_dims(train_x, axis=3)
        train_y = np.expand_dims(train_y, axis=3)
        model.fit(train_x, train_y, batch_size=16,epochs=20, validation_split=0.1, verbose=0)
    
    model.save('weights_3.h5') 
    files = os.listdir(train_images) 
    for i in files:
        print("270) ",i)
        train_x, train_y = load_train_set_con([i]) 
        train_x = np.rot90(train_x, 3)
        train_x = np.rot90(train_x, 3)
        train_x = np.rot90(train_x, 3)
        train_y = np.rot90(train_y, 3)
        train_y = np.rot90(train_y, 3)
        train_y = np.rot90(train_y, 3)
        train_x = cv2.resize(train_x, (448, 448)) 
        train_y = cv2.resize(train_y, (448, 448))
        train_x = blockshaped(train_x,112,112)
        train_y = blockshaped(train_y,112,112)
        train_x = np.expand_dims(train_x, axis=3)
        train_y = np.expand_dims(train_y, axis=3)
        model.fit(train_x, train_y, batch_size=16,epochs=20, validation_split=0.1, verbose=0)
    model.save('weights_4.h5') 
    
    
    


def train_image_full():
     
    model.add(Dense(500, input_shape=(29,),activation = 'tanh'))
    model.add(Dense(1, activation = 'tanh'))
    model.compile(loss='mean_absolute_error',
              optimizer='adam',
              metrics=['accuracy'])
    
    
    files = os.listdir(train_images_full) 
    for i in files:
        train_x, train_y = load_train_set_full([i])
        print(i)
        model.fit(train_x, train_y, batch_size=16,epochs=20, validation_split=0.1, verbose=1)
    
    model.save('weights_full.h5') 
    
    
    
    
def main():
    train_con()
    train_image_full()
    

if __name__ == '__main__':
    main()
