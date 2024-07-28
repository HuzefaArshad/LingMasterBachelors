import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle
word_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C',
             13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
             25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a',
             37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k', 47: 'l', 48: 'm',
             49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y',
             61: 'z'}

################ PARAMETERS ########################
FolderPath = 'Img'
copimg=0
Testingratiopercent = 0.2
Validationratiopercent = 0.2
StandardImageDimensions= (64,64,3)
ValidatioOfBatchSize= 50
Box_name = 'image'
IterEpochsVal = 2
PerStepEpochsVal= 2000
ListOfImages = []     # LIST CONTAINING ALL THE IMAGES
ListOFClassNo = []    # LIST CONTAINING ALL THE CORRESPONDING CLASS ID OF IMAGES
###############################################################################
DataList = os.listdir(FolderPath)
print("Total number of classes==>",len(DataList))
TotalNoClasses = len(DataList)
print("Classes will be importing==>")
for x in range (0,TotalNoClasses):
    ListOfPictures = os.listdir(FolderPath+"/"+str(x))
    for y in ListOfPictures:
        ImCurrentImg= cv2.imread(FolderPath+"/"+str(x)+"/"+y)            #LOADING DATA FROM DIRECTORY
        ImCurrentImg= cv2.resize(ImCurrentImg,(64,64))
        ListOfImages.append(ImCurrentImg)
        ListOFClassNo.append(x)
    print(x,end= " ")
##################################################################################
print(" ")
print("No of images==> ",len(ListOfImages))
print("No of ids in classlist===>= ",len(ListOFClassNo))
############################################################################
ListOfImages = np.array(ListOfImages)
ListOFClassNo = np.array(ListOFClassNo)                                                         #now converting list into array using numpy
print(ListOfImages.shape)
############################################################################
########################################################################################################
Var_Xtrain,Var_Xtest,var_ytrain,var_ytest = train_test_split(ListOfImages,ListOFClassNo,test_size=Testingratiopercent)                   #Now splitting the data into three portions test train validation
Var_Xtrain,X_validation,var_ytrain,y_validation = train_test_split(Var_Xtrain,var_ytrain,test_size=Validationratiopercent)       #first 20% will be given to test and rest of 80% will be given to
print(Var_Xtrain.shape)                                                                                   #train the we will again split train list and 20 % will be given
print(Var_Xtest.shape)                                                                                    # to validation and train will be left with only 60% we have performed
print(X_validation.shape)                                                                              #this task to shuffle the data so that we will not miss any single class
                                                                                                       #X unit denotes the actual data where as Y defines the class id of that data

#########################################################################################################
########################################################################################################
ListofSamples= []
for x in range(0,TotalNoClasses):
    # print(len(np.where(y_train==x)[0]))
    ListofSamples.append(len(np.where(var_ytrain==x)[0]))
# print(numOfSamples)                                     #now we will plot all sample of id`s present in Y_train using matplotlib
plt.figure(figsize=(10,5))
plt.bar(range(0,TotalNoClasses),ListofSamples)
plt.title("No of Images for each Class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()
##########################################################################################################
def Meth_preProcessingTask(Temp_img):
    Temp_img = cv2.cvtColor(Temp_img,cv2.COLOR_BGR2GRAY)
    Temp_img = cv2.equalizeHist(Temp_img)                         #method for preprocessing taking image as in input converting it into 1 channel and then balancing the image
    Temp_img = Temp_img/255                                       #we have divided the image with 255 so that it will be easy for us to classify the intensity of image betweeen 0-1

    return Temp_img
########################################################################################################
Var_Xtrain= np.array(list(map(Meth_preProcessingTask,Var_Xtrain)))
Var_Xtest= np.array(list(map(Meth_preProcessingTask,Var_Xtest)))                        #we have used map function to send all data in a single line of code in preprocessing function
X_validation= np.array(list(map(Meth_preProcessingTask,X_validation)))
########################################################################################################
##############################################################################################################################
Var_Xtrain = Var_Xtrain.reshape(Var_Xtrain.shape[0],Var_Xtrain.shape[1],Var_Xtrain.shape[2],1)
Var_Xtest = Var_Xtest.reshape(Var_Xtest.shape[0],Var_Xtest.shape[1],Var_Xtest.shape[2],1)                                        #reshaping image and adding the depth of 1 which is nessasary or
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)          #good when  we are using convolutional nueral network
################################################################################################################################
##############################################################################################################################
dataGen = ImageDataGenerator(Shifting_W_Range=0.1,
                             Shifting_H_Range=0.1,  #here we are performing image augumenatation using imagedatagenrator and we have defined the set of parameters
                             Range_zoom=0.2,         #so it will boost our dataset but it will only genrate images at the time of training
                             Range_Shear=0.1,
                             Range_Rotate=10)
dataGen.fit(Var_Xtrain)       #we are passing only dataset present in train variable
################################################################################################################################
EncodingLabel = LabelEncoder()
Tempvector1 = EncodingLabel.fit_transform(var_ytrain)
Tempvector2 = EncodingLabel.fit_transform(var_ytest)          #we are using label encoder so that we can use those list in our hot encoding of matrix
Tempvector3 = EncodingLabel.fit_transform(y_validation)
###################################################################################################################################
############################################################################################################################
var_ytrain = to_categorical(Tempvector1,TotalNoClasses)
var_ytest = to_categorical(Tempvector2,TotalNoClasses)           #here we are using one hot encoding of matrices which is nessasry for network or we can say per req
y_validation = to_categorical(Tempvector3,TotalNoClasses)
############################################################################################################################
def Meth_Model_Specs():
    TotalNoFilters = 60
    size_1_Filter = (5,5)
    size_2_Filter = (3, 3)
    Size_Pool = (2,2)
    Total_nodes= 500
    Ocrmodel = Sequential()   #we are using leenet archetecture
    Ocrmodel.add((Conv2D(TotalNoFilters,size_1_Filter,input_shape=(StandardImageDimensions[0],
                      StandardImageDimensions[1],1),activation='relu')))
    Ocrmodel.add((Conv2D(TotalNoFilters,size_1_Filter, activation='relu')))
    Ocrmodel.add(MaxPooling2D(pool_size=Size_Pool))
    Ocrmodel.add((Conv2D(TotalNoFilters//2, size_2_Filter, activation='relu')))
    Ocrmodel.add((Conv2D(TotalNoFilters//2, size_2_Filter, activation='relu')))
    Ocrmodel.add(MaxPooling2D(pool_size=Size_Pool))
    Ocrmodel.add(Dropout(0.5))
    Ocrmodel.add(Flatten())
    Ocrmodel.add(Dense(Total_nodes,activation='relu'))
    Ocrmodel.add(Dropout(0.5))
    Ocrmodel.add(Dense(TotalNoClasses, activation='softmax'))
    Ocrmodel.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return Ocrmodel
Ocrmodel = Meth_Model_Specs()
print(Ocrmodel.summary())
##########################################################################################################################
##########################################################################################################################
Model_History = Ocrmodel.fit_generator(dataGen.flow(Var_Xtrain,var_ytrain,
                                 batch_size=ValidatioOfBatchSize),
                                 steps_per_epoch=PerStepEpochsVal,  #starting the training process from here and the data augumentation that we have written previously will
                                 epochs=IterEpochsVal,                  #genrate the augumented data set at teh time of tarining
                                 validation_data=(X_validation,y_validation),
                                 shuffle=1)
###########################################################################################################################
#### PLOT THE RESULTS
plt.figure(1)
plt.plot(Model_History.history['loss'])
plt.plot(Model_History.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(Model_History.history['accuracy'])
plt.plot(Model_History.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
#### EVALUATE USING TEST IMAGES
TestScore = Ocrmodel.evaluate(Var_Xtest,var_ytest,verbose=0)
print('Each Test Score==> ',TestScore[0])
print('Each Test Accuracy==>', TestScore[1])
#### SAVE THE TRAINED MODEL
pickle_out= open("model_trained.p", "wb")
pickle.dump(Ocrmodel,pickle_out)
pickle_out.close()


