from PIL import Image
import numpy as np
import os
pth1='C:/Users/Pranesh/Desktop/images'
pth2='C:/Users/Pranesh/Desktop/another1'

os.mkdir(pth2)

def get_imlist(path):
    return [os.path.join(f) for f in os.listdir(path) if f.endswith('.tif')]


listing=get_imlist(pth1)
q=np.floor(np.random.rand(10000)*37000)
q=q.astype('int32')
listingb=[]
for i in q:
    listingb.append(listing[i])
    
for file in listingb:
    im=Image.open(pth1+'\\'+file)
    img=im.resize((150,150))
    img.save(pth2+'\\'+file)


j=0
for i in plisting:
    ps=i.index('_')
    s=int(i[1:ps])
    if s==7:
        label[j]=.277
    elif s==8:
        label[j]=.404
    elif s==9:
        label[j]=.182
    elif s==10:
        label[j]=.193
    elif s==11:
        label[j]=.270
    elif s==12:
        label[j]=.451
    elif s==13:
        label[j]=.228
    elif s==14:
        label[j]=.256
    elif s==15:
        label[j]=.366
    elif s==16:
        label[j]=.505
    elif s==17:
        label[j]=.184
    elif s==18:
        label[j]=.224
    j+=1

b=30000
c=10000
t=0
while t<80000:
    
    if t%10000==0:
        q=np.floor(np.random.rand(100000,1)*b)
        w=np.floor(np.random.rand(100000,1)*c)
   
    if w[t]<37000 and q[t]<37000:
            intit=all_img[int(w[t])]
            all_img[int(w[t])]=all_img[int(q[t])]
            all_img[int(q[t])]=intit
            intit=label[int(w[t])]
            label[int(w[t])]=label[int(q[t])]
            label[int(q[t])]=intit
    t+=1
    if t%5000==0:
        b=b*.5
        c=c*.5

im_test=all_img[12000:]
im_train=all_img[:12000]
te_lab=label[12000:]
tr_lab=label[:12000]

from keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(rescale=None)

from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def custom_activation(x):
    return (K.sigmoid(x) * 2) - 1
def cs_elu(x):
    return K.elu(x)*.577


from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
model=models.Sequential()
model.add(layers.Conv2D(6,(7,7),activation='elu',input_shape=(101,300,1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(18, (3,3), activation=cs_elu))
model.add(layers.MaxPooling2D((2, 2)))


model.add(layers.Conv2D(26, (3, 3), activation=cs_elu))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(44, (3, 3), activation=cs_elu))

model.add(layers.Conv2D(70, (3, 3), activation='elu'))



model.add(layers.Conv2D(128, (3, 3), activation=cs_elu))

model.add(layers.Conv2D(300, (3, 3), activation=cs_elu))
model.add(layers.Dropout(.7))

model.fit_generator(datagen.flow(all_img,label, batch_size=150),steps_per_epoch=len(all_img) / 32, epochs=2)



model.add(layers.Conv2D(256,(1,1),activation=cs_elu))
model.add(layers.Conv2D(90,(1,14),activation=custom_activation))
model.add(layers.Flatten())
model.add(layers.Dense(1))
model.compile(loss='mse',optimizer=optimizers.Adam(lr=.0001,beta_1=0.9,beta_2=.999),metrics=['mae'])


