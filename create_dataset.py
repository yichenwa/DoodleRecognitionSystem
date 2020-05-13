import os
import numpy as np
import random

my_trainset = []
my_trainlabel = []
my_testset = []
my_testlabel = []
index = 0
for filename in os.listdir('dataset'):
    path = 'dataset/' + filename
    dataset = np.load(path)
    for i in range(0,20000):
        my_trainset.append(dataset[i])
        my_trainlabel.append(index)
    for j in range(20000,22000):
        my_testset.append(dataset[j])
        my_testlabel.append(index)
    index += 1
shuffle_combine = list(zip(my_trainset,my_trainlabel))
random.shuffle(shuffle_combine)
my_trainset, my_trainlabel = zip(*shuffle_combine)

shuffle_combine_test = list(zip(my_testset,my_testlabel))
random.shuffle(shuffle_combine_test)
my_testset, my_testlabel = zip(*shuffle_combine_test)

my_trainlabel = np.asarray(my_trainlabel)
np.save('yic_train_image',my_trainset)
np.save('yic_train_label', my_trainlabel)

my_testlabel = np.asarray(my_testlabel)
np.save('yic_test_image', my_testset)
np.save('yic_test_label', my_testlabel)