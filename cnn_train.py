import numpy as np
import itertools
import cv2
import os
import h5py
import re

from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

#################### set parameters ######################

path = '../path/to/data'
wx = 13
wy = 2

#################### create dataset ######################

def create_data(path,wx,wy):
    X = []
    y = []
    for roots, dirs, files in os.walk(path):
        #files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        files.sort(key=alphanum_key)

        file = files[0::2]
        gt_files = files[1::2]
        file = file[:70]
        gt_files = gt_files[:70]

        Xp, Xn, yp, yn = ([] for i in range(4))

        for (f,g) in zip(file,gt_files):
            fname  = os.path.join(roots,f)
            gtname = os.path.join(roots,g)
            img = cv2.imread(fname)
            gt  = cv2.imread(gtname)
            h, w, d = img.shape
            # padding to borders of image
            pimg = cv2.copyMakeBorder(img,wx,wx,wx,wx,cv2.BORDER_REFLECT101)
            pgt  = cv2.copyMakeBorder(gt,wy,wy,wy,wy,cv2.BORDER_REFLECT101)

            cnt = 0
            for i in range(wx,w+wx):
                for j in range(wx,h+wx):
                    if (pgt[j-wx+wy,i-wx+wy,0]==0): # positive sample (black pixel)
                        patch = pimg[j-wx:j+wx+1,i-wx:i+wx+1,:]
                        label = pgt[j-wx:j-wx+2*wy+1,i-wx:i-wx+2*wy+1,0]
                        label = np.reshape(label,((2*wy+1)**2,),order='F')

                        Xp.append(patch)
                        yp.append(label)
                        cnt += 1

            all_idx = list(itertools.product(range(wx,w+wx), range(wx,h+wx)))
            np.random.shuffle(all_idx)
            neg_idx = all_idx[:2*cnt]

            for (i,j) in neg_idx:
                if (pgt[j-wx+wy,i-wx+wy,0]==255):  # negative sample (white pixel)
                    patch = pimg[j-wx:j+wx+1,i-wx:i+wx+1,:]
                    label = pgt[j-wx:j-wx+2*wy+1,i-wx:i-wx+2*wy+1,0]
                    label = np.reshape(label,((2*wy+1)**2,),order='F')

                    Xn.append(patch)
                    yn.append(label)

        X = np.vstack((Xp,Xn))
        y = np.vstack((yp,yn))

    return np.array(X), np.array(y)

X, y = create_data(path,wx,wy)

print(np.shape(X))
print(np.shape(y))

X = X.astype(np.float32)
y = y.astype(np.float32)
X /= 255.0
y /= 255.0
print("data normalized")

################### train CNN model ########################
def cnn_model():
    model = Sequential()
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

    return model

# train/validation split
Xtrn, Xtst, ytrn, ytst = train_test_split(X,y,test_size=0.2,shuffle=True,random_state=42)

early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
mdl_save = ModelCheckpoint('mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='auto')


model = cnn_model()
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
