import numpy as np
import cv2
import scipy.misc

from keras.models import model_from_json

############### set parameters #################
wx = 13
wy = 2
mode = 1 # 1 for CNN/ 0 for HOG

############### load model ######################

if mode:
    json_file = open('model_cnn.json', 'r')
else:
    json_file = open('model_hog.json', 'r')

model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

# load weights into new model
if mode:
    model.load_weights("mdl_wts_cnn.hdf5")
else:
    model.load_weights("mdl_wts_hog.hdf5")

model.compile(loss='binary_crossentropy', optimizer='adam')
print("Loaded model from disk")

############### test on unseen data ######################

filename = 'testimg.jpg'
tst_img = cv2.imread(filename)
tst_img = tst_img.astype(np.float32)/255
#tst_img = cv2.resize(tst_img,(200,200),cv2.INTER_CUBIC)
pimg = cv2.copyMakeBorder(tst_img,wx,wx,wx,wx,cv2.BORDER_REFLECT101)
h, w, d = tst_img.shape

img_out = np.zeros((h,w), dtype=tst_img.dtype)
img_out = cv2.copyMakeBorder(img_out,wy,wy,wy,wy,cv2.BORDER_REFLECT101)

for i in range(wx,w+wx):
    for j in range(wx,h+wx):
        Xtst = pimg[j-wx:j+wx+1,i-wx:i+wx+1,:]
        Xtst = Xtst[np.newaxis,:,:,:]
        preds = model.predict(Xtst)
        
        label = np.reshape(preds,(2*wy+1,2*wy+1),order='F')
        
        img_out[j-wx:j-wx+2*wy+1,i-wx:i-wx+2*wy+1] = img_out[j-wx:j-wx+2*wy+1,i-wx:i-wx+2*wy+1] + label

# normalize image to get probabilty map [0,1]
img_out = img_out/img_out.max()

# thresholding binarization
img_out[img_out>=0.5] = 1
img_out[img_out<0.5] = 0

img_out = 255*img_out

# save predicted image 
scipy.misc.imsave('outfile1.jpg', img_out)
