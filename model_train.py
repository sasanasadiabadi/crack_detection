import numpy as np
import cv2
import os
import re

from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from skimage import feature 

#################### set parameters ######################

path = '../path/to/data'
wx = 13
wy = 2
mode = 0 # 1: CNN / 0: HOG

#################### create dataset ######################

def create_data(path,wx,wy,mode):
    X = []
    y = []
    
    files = os.listdir(path)
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    files.sort(key=alphanum_key)

    file = files[0::2]
    gt_files = files[1::2]
    
    Xp, Xn, yp, yn = ([] for i in range(4))

    for (f,g) in zip(file,gt_files):
        fname  = os.path.join(roots,f)
        gtname = os.path.join(roots,g)
        
        img = cv2.imread(fname).astype(np.float32)/255
        gt  = cv2.imread(gtname).astype(np.float32)/255
        
        h, w, d = img.shape
        # padding to borders of image
        pimg = cv2.copyMakeBorder(img,wx,wx,wx,wx,cv2.BORDER_REFLECT101)
        pgt  = cv2.copyMakeBorder(gt,wy,wy,wy,wy,cv2.BORDER_REFLECT101)

        # collect positive samples 
        cnt = 0
        for i in range(wx,w+wx):
            for j in range(wx,h+wx):
                if (pgt[j-wx+wy,i-wx+wy,0]==0): # positive sample (black pixel)
                    patch = pimg[j-wx:j+wx+1,i-wx:i+wx+1,:]

                    if mode: # CNN feature extraction
                        label = pgt[j-wx:j-wx+2*wy+1,i-wx:i-wx+2*wy+1,0]
                        label = np.reshape(label,((2*wy+1)**2,),order='F')

                        Xp.append(patch)
                        yp.append(label)

                    else: # HOG feature extraction
                        hg = feature.hog(patch[:,:,0], orientations=9, pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
                        label = 0

                        Xp.append(hg)
                        yp.append(label)
                    
                    cnt += 1

        # collect random negative samples 
        for i in np.random.choice(range(wx,w+wx),int(np.sqrt(2*cnt))):
            for j in np.random.choice(range(wx,h+wx),int(np.sqrt(2*cnt))):
                if (pgt[j-wx+wy,i-wx+wy,0]==255):  # negative sample (white pixel)
                    patch = pimg[j-wx:j+wx+1,i-wx:i+wx+1,:]
                    
                    if mode: # CNN feature extraction
                        label = pgt[j-wx:j-wx+2*wy+1,i-wx:i-wx+2*wy+1,0]
                        label = np.reshape(label,((2*wy+1)**2,),order='F')

                        Xn.append(patch)
                        yn.append(label)

                    else: # HOG feature extraction
                        hg = feature.hog(patch[:,:,0], orientations=9, pixels_per_cell=(8, 8),
                                         cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
                        label = 1

                        Xn.append(hg)
                        yn.append(label)

    X = np.vstack((Xp,Xn))
    y = np.concatenate((yp,yn),axis=0)

    return np.array(X), np.array(y)

X, y = create_data(path,wx,wy,mode)

print(np.shape(X))
print(np.shape(y))

################### train CNN model ########################
def train_model(mode):
    model = Sequential()
    
    if mode:
        # conv1
        model.add(Conv2D(128,kernel_size=(7,7),padding='same',activation='relu',input_shape=(2*wx+1,2*wx+1,3)))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        # conv2
        model.add(Conv2D(256,kernel_size=(5,5),padding='same',activation='relu'))
        model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

        model.add(Flatten())
        # dense1
        model.add(Dense(1000,activation='relu'))
        model.add(Dropout(0.5))
        # dense2
        model.add(Dense(1000, activation='relu'))
        model.add(Dropout(0.5))

        # ouput layer (multi-label classification)
        model.add(Dense((2*wy+1)**2,activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
    
    else:
        model.add(Dense(512, activation='relu', input_dim=np.shape(X)[1]))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(512, activation='relu'))

        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# train/validation split
Xtrn, Xtst, ytrn, ytst = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=42)

early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
mdl_save = ModelCheckpoint('mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='auto')


model = train_model(mode)
model.fit(Xtrn,ytrn,batch_size=128,epochs=10,shuffle=True,verbose=2,
          validation_data=(Xtst,ytst),callbacks=[early_stop,mdl_save])

### save model
file = os.path.expanduser('~') + '../path/to/directory/model.json'
try:
    os.remove(file)
except OSError:
    pass

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

print("Model saved to disk")
