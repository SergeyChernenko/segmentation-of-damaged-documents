import numpy as np
import cv2
import os
import itertools
import pytesseract
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, UpSampling2D, Flatten
from keras import backend as K
from keras.optimizers import Adam
from PIL import Image

model = Sequential()
src_path = os.path.dirname(__file__)+'\\'

im_scan = ""

image_sc = ".png" 


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
    

def target_matrix_full(img):
       
    return (img / 255.0).astype('float32').ravel()[:, None]

def blockshaped(arr, nrows, ncols): # разрезка изображения
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
    model.load_weights('weights_con.h5')
    test_image_start = load_image(os.path.join(im_scan)) 
    test_x = feature_matrix_con(test_image_start)
    test_image = cv2.resize(test_image_start, (448, 448)) 
    test_x = cv2.resize(test_x, (448, 448))
    test_x = blockshaped(test_x,112,112)
    test_x = np.expand_dims(test_x, axis=3)
    
    y_pred = model.predict_on_batch(test_x)
    nm=[]
    nm1=[]
    nm2=[]
    nm3=[]
    nm4=[]
    nm.append(np.hstack((y_pred[0],y_pred[1],y_pred[2],y_pred[3])))
    nm1.append(np.hstack((y_pred[4],y_pred[5],y_pred[6],y_pred[7])))
    nm2.append(np.hstack((y_pred[8],y_pred[9],y_pred[10],y_pred[11])))
    nm3.append(np.hstack((y_pred[12],y_pred[13],y_pred[14],y_pred[15])))
    nm4 = np.hstack((nm, nm1,nm2,nm3)) 
    
    output_test = nm4.reshape(nm4.shape)*255.0 
    output_test = output_test.reshape(448,448)
    output_test = cv2.resize(output_test, (test_image_start.shape[1], test_image_start.shape[0]))
          
    cv2.imwrite('con_1' + image_sc, output_test)
    
   
def ful_s():
        
    model.add(Dense(500, input_shape=(29,),activation = 'tanh'))
    model.add(Dense(1, activation = 'tanh'))
    model.compile(loss='mean_absolute_error',
              optimizer='adam',
              metrics=['accuracy'])
    


def train_image_full():
       
    model.load_weights('weights_full.h5')
    test_image_con = load_image(src_path, "con_1.png")
    test_x = feature_matrix_full(test_image_con)
    y_pred = model.predict_on_batch(test_x)
    output = y_pred.reshape(test_image_con.shape)*255.0 
    output = cv2.resize(output, (test_image_con.shape[1], test_image_con.shape[0]))
    
    cv2.imwrite('final' + image_sc, output)
    
     
def get_string(img_path):
    
    img = cv2.imread(img_path)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    result = pytesseract.image_to_string(Image.open(img_path))
    return result

def recognition():
    text = get_string(src_path + "final.png")
    file = open ("testfile.txt", "w")  
    file.write (str(text.encode("utf-8")))
    file.close
    file = open("testfile.txt", "r")
    os.startfile(r'testfile.txt')
    
    
def main():
    
    train_con()
    ful_s()
    train_image_full()
    recognition()
      

if __name__ == '__main__':
    main()