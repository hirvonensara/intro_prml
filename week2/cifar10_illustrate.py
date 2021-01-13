"""
Introduction to Pattern Recognition and Machine Learning
Exercise 2 Visual Classification
Sara Hirvonen 283839
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
import time

def unpickle(file):
    with open(file, 'rb') as f:
        dct = pickle.load(f, encoding="latin1")
    return dct

""" Takes two numpy arrays as parameters and
compares how many same elements they posses,
returns the similarity as %  """
def class_acc(pred, gt):
    numerator = 0
    denominator = len(gt)

    for i in range(len(gt)):
        if (pred[i]==gt[i]):
            numerator += 1

    accuracy = (numerator/denominator)*100
    return accuracy

def cifar10_classifier_random(x):
    random_labels = np.zeros(len(x), dtype=int)

    for i in range(len(x)):
        random_labels[i]= np.random.randint(0,10)

    return random_labels

""" First version of the 1-NN classifier, not used
in the final code as it's not as efficient"""
def cifar10_classifier_1nn_v1(x,trdata,trlabels):
    x = x.astype('int32')
    trdata = trdata.astype('int32')
    distance_arr = np.array([])
    index = 0
    for vector in trdata:
        vector = vector.astype('int32')
        sum= 0
        for i in range(len(vector)):
            ds = vector[i]-x[i]
            sum = sum + pow(ds, 2)
        distance_arr = np.append(distance_arr, sum)
        index += 1

    smallest_distance = np.argmin(distance_arr)
    return trlabels[smallest_distance]

def cifar10_classifier_1nn_v2(x,trdata,trlabels):
    x = x.astype('int32')
    trdata = trdata.astype('int32')
    distance_arr = np.array([])
    index = 0
    for vector in trdata:
        dis = np.sum(pow(np.subtract(vector,x),2))
        distance_arr=np.append(distance_arr,dis)

    smallest_distance = np.argmin(distance_arr)
    return trlabels[smallest_distance]

def get_time(start_time):
    time_str = ""
    time_struct= time.gmtime(time.time()-start_time)

    if time_struct.tm_hour > 0:
        time_str = time_str + str(time_struct.tm_hour) + " hours "
    if time_struct.tm_min > 0:
        time_str = time_str + str(time_struct.tm_min) + " minutes "
    if time_struct.tm_sec < 1:
        time_str = str(time.time()-start_time) + " seconds"
    if time_struct.tm_sec > 1:
        time_str = time_str + str(time_struct.tm_sec) +" seconds "

    return time_str

def main():
    start_time = time.time()
    datadict_1 = unpickle('C:/Users/sara_/OneDrive/Asiakirjat/koodi/cifar10/cifar-10-batches-py/data_batch_1')
    datadict_2 = unpickle('C:/Users/sara_/OneDrive/Asiakirjat/koodi/cifar10/cifar-10-batches-py/data_batch_2')
    datadict_3 = unpickle('C:/Users/sara_/OneDrive/Asiakirjat/koodi/cifar10/cifar-10-batches-py/data_batch_3')
    datadict_4 = unpickle('C:/Users/sara_/OneDrive/Asiakirjat/koodi/cifar10/cifar-10-batches-py/data_batch_4')
    datadict_5 = unpickle('C:/Users/sara_/OneDrive/Asiakirjat/koodi/cifar10/cifar-10-batches-py/data_batch_5')

    test_datadict = unpickle('C:/Users/sara_/OneDrive/Asiakirjat/koodi/cifar10/cifar-10-batches-py/test_batch')

    X1 = datadict_1["data"]
    Y1 = datadict_1["labels"]
    X2 = datadict_2["data"]
    Y2 = datadict_2["labels"]
    X3 = datadict_3["data"]
    Y3 = datadict_3["labels"]
    X4 = datadict_4["data"]
    Y4 = datadict_4["labels"]
    X5 = datadict_5["data"]
    Y5 = datadict_5["labels"]

    training_data = np.concatenate((X1,X2,X3,X4,X5), axis =0)
    training_labels = np.concatenate((Y1,Y2,Y3,Y4,Y5), axis=0)

    test_data = test_datadict["data"]
    test_labels  = test_datadict["labels"]

    """
    N = 100
    X = X1[0:N:1]
    Y= Y1[0:N:1]
    test_data = test_data[0:N:1]
    test_labels = test_labels[0:N:1]
    

    # For testing if the 100% accuracy works
    test_data = X
    """

    labeldict = unpickle('C:/Users/sara_/OneDrive/Asiakirjat/koodi/cifar10/cifar-10-batches-py/batches.meta')
    label_names = labeldict["label_names"]

    

    # tässä välissä 1-N-N classifier
    labels_1nn = np.zeros(len(test_labels), dtype=int)
    i = 0
    for test_sample in test_data:
        label = cifar10_classifier_1nn_v2(test_sample, training_data, training_labels)
        labels_1nn[i]=label
        i += 1

        if (i%1000==0):
            print("Time so far is", get_time(start_time))

    
    #random_labels = cifar10_classifier_random(training_data)
    accuracy_1nn = class_acc(labels_1nn, test_labels)
    #random_accuracy = class_acc(random_labels, training_labels)
    #print("Random accuracy is", random_accuracy, "%")
    print("Accuracy is", accuracy_1nn, "%")
    total_time = get_time(start_time)
    print("The total time is", total_time)
    


"""
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    Y = np.array(Y)


    for i in range(X.shape[0]):
        # Show some images randomly
        if random() > 0.999:
            plt.figure(1)
            plt.clf()
            plt.imshow(X[i])
            plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
            plt.pause(1)
"""
main()
