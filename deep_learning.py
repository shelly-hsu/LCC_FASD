import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np 
from PIL import Image 
from tensorflow.keras.applications.vgg16 import VGG16
from keras.models import Sequential 
from keras.models import load_model 
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D 
from keras.utils import np_utils 
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 

train_path = ".\\LCC_FASD_training_2"
validation_path = ".\\LCC_FASD_validation" 
test_path = ".\\LCC_FASD_test" 



#通過ImageDataGenerator來生成一個數據生成器，其中rescale引數指定將影象張量的數字縮放。
train_data = ImageDataGenerator(rescale = 1./255)
validation_data = ImageDataGenerator(rescale = 1./255)
test_data = ImageDataGenerator(rescale = 1./255)

#batch_size：一次性讀入多少批量的圖片。
train_generator = train_data.flow_from_directory(train_path,   #目標目錄
                                                target_size=(224,224),  #所有影像調整為168*168
                                                batch_size=32, 
                                                shuffle = True,   #混洗數據
                                                class_mode='categorical')
validation_generator = validation_data.flow_from_directory(validation_path,
                                                           target_size=(224,224),
                                                           batch_size=32,
                                                           shuffle=False,
                                                          class_mode='categorical')
test_generator = test_data.flow_from_directory(test_path, target_size=(224,224),batch_size=32, shuffle=False, class_mode=None)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=2, activation='softmax'))

#compile model          
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy']) 

train_steps_per_epoch = np.ceil(15879/32) #batch_size=32
validation_steps_per_epoch = np.ceil((7580/32))
test_steps_per_epoch = np.ceil((2948/32))

train_generator.reset()
#使用批量生成器擬和模型逐批生成的數據，按此批次訓練模型
train_history=model.fit_generator(generator=train_generator,
                        steps_per_epoch = train_steps_per_epoch,  #訓練多少輪次epochs
                        epochs=10,   #數據迭代的輪數
                        validation_data = validation_generator,#傳入validation_data。當validation_data為生成器時，本參數指定驗證集的生成器返回次數
                        validation_steps =validation_steps_per_epoch,
                        workers=1,   #並行生成批處理的線程數
                        use_multiprocessing=False)


validation_generator.reset()
#在數據稱成器上評估模型
score = model.evaluate_generator(generator=validation_generator,
                                 steps=validation_steps_per_epoch,
                                 max_queue_size=10, 
                                 workers=1, 
                                 use_multiprocessing=False,
                                 verbose=1)
print('Test loss:', score[0])
print('Test accurancy:', score[1])

plt.subplot(121)
plt.plot(train_history.history['accuracy'])
plt.plot(train_history.history['val_accuracy'])
plt.title('Train History')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Test'],loc='upper left')

plt.subplot(122)
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.title('Train History')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train','Test'],loc='upper left')

plt.show()


predicted_class_indices=np.argmax(predict,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k, v in labels.items())
predictions=[]
fileNames = []
for k in predicted_class_indices:
    if labels[k] == 'real':
        predictions.append(1)
    elif labels[k] == 'spoof':
        predictions.append(0)
for root, dirs, files in os.walk(test_path):
     for f in files:
        fileNames.append(f)
result=pd.DataFrame({"FileName":fileNames,
                    "Label(Real:1, Spoof:0)":predictions})
result.to_csv("LCC_FASD_test_Label_submmit.csv",index=False)