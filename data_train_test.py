import tensorflow as tf
import os
import numpy as np
import cv2


def getting_data():
#Collecting data to train

    k = os.listdir("data")

    x_1 = []
    x_2 = []
    y_data = []
    #same input and validation person
    for roll_no in k:
        path = os.path.join(r"data",roll_no)
        images = os.listdir(path)
        if(len(images)>10):
            random_images = np.random.choice(images,15,replace = False)
        else:
            random_images = np.random.choice(images,len(images),replace = False)
        for i in range(len(random_images)):
            temp1 = cv2.imread(os.path.join(path,random_images[i]))
            temp1 = cv2.cvtColor(temp1,cv2.COLOR_BGR2GRAY)
            for j in range(i,len(random_images)):
                temp2 = cv2.imread(os.path.join(path,random_images[j]))
                temp2 = cv2.cvtColor(temp2,cv2.COLOR_BGR2GRAY)
                x_1.append(temp1/255.0)
                x_2.append(temp2/255.0)
                y_data.append(1.0)

    

    #negative input and validation person
    for i in range(len(k)):
        img1_path = os.path.join(r"data",k[i])
        imgs1_lis = np.random.choice(os.listdir(img1_path),12,replace=False)
        imgs1 = []
        imgs2 = []
        for f in imgs1_lis:
            t1 = cv2.imread(os.path.join(img1_path,f))
            imgs1.append(cv2.cvtColor(t1,cv2.COLOR_BGR2GRAY))
        for j in range(len(k)):
            if(i == j ):
                continue
            img2_path = os.path.join(r"data",k[j])
            imgs2_lis = np.random.choice(os.listdir(img2_path),3,replace=False)
            for s in imgs2_lis:
                t2 = cv2.imread(os.path.join(img2_path,s))
                imgs2.append(cv2.cvtColor(t2,cv2.COLOR_BGR2GRAY))
        for temp1 in imgs1:
            for temp2 in imgs2:
                x_1.append(temp1/255.0)
                x_2.append(temp2/255.0)
                y_data.append(0.0)

    #making use of tensorflow pipeline for training model batch wise
    tf_data = tf.data.Dataset.from_tensor_slices((x_1,x_2,y_data))
    data = tf_data.shuffle(buffer_size=1024)

    #spliting traing and testing data

    #trainig data batches
    train_data = data.take(round(len(data)*0.7))
    train_data = train_data.batch(16)
    train_data = train_data.prefetch(8)

    #testing data batches
    test_data = data.skip(round(len(data)*0.7))
    test_data = test_data.take(round(len(data)*0.3))
    test_data = test_data.batch(16)
    test_data = test_data.prefetch(8)

    return train_data,test_data








