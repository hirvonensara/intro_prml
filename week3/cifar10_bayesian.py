"""
Introduction to Pattern Recognition and Machine Learning
Exercise 3 Visual Classification
Sara Hirvonen
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.stats
from random import random
from scipy.stats import norm
from skimage.transform import rescale, resize, downscale_local_mean

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

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

""" Converts the original 50000X32X32X3 images
in X to Xf (50000x3)"""
def cifar10_color(X):
    N = X.shape[0]
    Xf = np.zeros((N,3))

    for i in range(X.shape[0]):
        img = X[i]
        img_1x1 = resize(img, (1, 1))     
        r_vals = img_1x1[:,:,0].reshape(1*1)
        g_vals = img_1x1[:,:,1].reshape(1*1)
        b_vals = img_1x1[:,:,2].reshape(1*1)
        mu_r = r_vals.mean()
        mu_g = g_vals.mean()
        mu_b = b_vals.mean()
        Xf[i,:] = (mu_r, mu_g, mu_b)
    
    return Xf

""" Resizes original 50000x32x32x3 images 
to 2x2 images and returns Xf (50000x12)"""
def cifar10_2x2_color(X):
    N = X.shape[0]
    Xf = np.zeros((N,12))

    for i in range(X.shape[0]):
        img = X[i]
        img_2x2 = resize(img, (2, 2))  
        Xf[i,:] = np.concatenate((img_2x2[0][0],img_2x2[0][1], img_2x2[1][0], img_2x2[1][1]), axis=0)
    
    return Xf

""" Returns mean (mu), standard deviation (sigma)
and probalities (p) for all classes"""
def cifar10_naivebayes_learn(Xf, Y):
    size = Xf.shape[1]
    mu = np.zeros((10,size))
    sigma = np.zeros((10,size))
    p = np.full((10,1), 0.1)

    classes = [[] for _ in range(10)]

    for i, data in enumerate(Xf):
        classes[Y[i]].append(data)

    classes = np.asarray(classes, dtype=object)

    for i, images in enumerate(classes):
        mu[i] = np.mean(images, axis=0)
        sigma[i] = np.std(images, axis=0, dtype = np.float64)
    return mu, sigma, p

""" Naive bayesian classifier computes the
probabilities of each class and returns the
label that gives the highest probability"""
def cifar10_classifier_naivebayes(x, mu, sigma, p):
    all_probs = np.zeros(10)

    for i in range(10):
        prob = norm.pdf(x[0],mu[i][0],sigma[i][0])*norm.pdf(x[1],mu[i][1],sigma[i][1])*norm.pdf(x[2],mu[i][2],sigma[i][2])*p[i]
        all_probs[i]=prob

    label = np.argmax(all_probs)
    return label

""" Returns mean, covariance and probability 
matrixes for all classes """
def cifar10_bayes_learn(Xf, Y):
    size = Xf.shape[1]
    print(size)
    mu = np.zeros((10,size))
    sigma = np.zeros((10,size,size))
    p = np.full((10,1), 0.1)

    classes = [[] for _ in range(10)]

    for i, data in enumerate(Xf):
        classes[Y[i]].append(data)

    classes = np.asarray(classes, dtype=object)

    for i, images in enumerate(classes):
        images = images.astype(np.float64)
        mu[i] = np.mean(images, axis=0)
        sigma[i] = np.cov(images, rowvar=False)
    return mu, sigma, p

""" Bayesian classifier computes the probabilities
of each class for an image x and return the label
which has the highest probability"""
def cifar10_classifier_bayes(x, mu, sigma, p):
    all_probs = np.zeros(10)

    for i in range(10):
        prob = scipy.stats.multivariate_normal(mu[i], sigma[i]).pdf(x)*p[i]
        all_probs[i]=prob

    label = np.argmax(all_probs)
    return label

""" Gives time as string """
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

    X = np.concatenate((X1,X2,X3,X4,X5), axis =0)
    Y = np.concatenate((Y1,Y2,Y3,Y4,Y5), axis=0)

    test_datadict = unpickle('C:/Users/sara_/OneDrive/Asiakirjat/koodi/cifar10/cifar-10-batches-py/test_batch')

    test_data = test_datadict["data"]
    test_labels = test_datadict["labels"]

    """
    labeldict = unpickle('C:/Users/sara_/OneDrive/Asiakirjat/koodi/cifar10/cifar-10-batches-py/batches.meta')
    label_names = labeldict["label_names"]
    """
    X = X.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")

    Y = np.array(Y)
    Xf = cifar10_color(X)
    Xf_2x2 = cifar10_2x2_color(X)

    test_data = test_data.reshape(10000,3,32,32).transpose(0,2,3,1).astype("uint8")
    test_X = cifar10_color(test_data)
    test_X_2x2 = cifar10_2x2_color(test_data)

    mu_naive, sigma_naive, p1 = cifar10_naivebayes_learn(Xf, Y)
    mu_bayes, sigma_bayes, p2 = cifar10_bayes_learn(Xf, Y) 
    mu_bayes_2x2, sigma_bayes_2x2, p2_2x2 = cifar10_bayes_learn(Xf_2x2,Y)
    print("Learning done, took", get_time(start_time))

    predicted_labels_naive = np.zeros(len(test_labels), dtype=int)
    predicted_labels_bayes = np.zeros(len(test_labels), dtype=int)
    predicted_labels_bayes_2x2 = np.zeros(len(test_labels), dtype=int)

    print("Started testing 1x1 images (naive bayesian and bayesian)")
    for i, test_img in enumerate(test_X):
        naive_label = cifar10_classifier_naivebayes(test_img, mu_naive, sigma_naive, p1)
        predicted_labels_naive[i] = naive_label
        bayes_label = cifar10_classifier_bayes(test_img, mu_bayes, sigma_bayes,p2)
        predicted_labels_bayes[i] = bayes_label

        if (i%5000==0):
            print("Time so far is", get_time(start_time))

    print("Started testing 2x2 images (bayesian)")
    for i, test_img in enumerate(test_X_2x2):
        bayes_label_2x2 = cifar10_classifier_bayes(test_img,mu_bayes_2x2, sigma_bayes_2x2,p2_2x2)
        predicted_labels_bayes_2x2[i] = bayes_label_2x2

        if (i%5000==0):
            print("Time so far is", get_time(start_time))
    print("Done calculating")
    accuracy_naive_bayes = class_acc(predicted_labels_naive, test_labels)
    accuracy_bayes = class_acc(predicted_labels_bayes, test_labels)
    accuracy_bayes_2x2 = class_acc(predicted_labels_bayes_2x2, test_labels)
    print("Naive Bayesian accuracy is ", accuracy_naive_bayes, "%")
    print("Bayesian accuracy for 1x1 images is ", accuracy_bayes, "%")
    print("Bayesian accuracy for 2x2 images is ", accuracy_bayes_2x2, "%")
    total_time = get_time(start_time)
    print("The total time is", total_time)

    """
    X_mean = np.zeros((10000,3))
    for i in range(X.shape[0]):
        # Convert images to mean values of each color channel
        img = X[i]
        img_8x8 = resize(img, (8, 8))        
        img_1x1 = resize(img, (1, 1))        
        r_vals = img_1x1[:,:,0].reshape(1*1)
        g_vals = img_1x1[:,:,1].reshape(1*1)
        b_vals = img_1x1[:,:,2].reshape(1*1)
        mu_r = r_vals.mean()
        mu_g = g_vals.mean()
        mu_b = b_vals.mean()
        X_mean[i,:] = (mu_r, mu_g, mu_b)
        
        # Show some images randomly
        if random() > 0.999:
            plt.figure(1);
            plt.clf()
            plt.imshow(img_8x8)
            plt.title(f"Image {i} label={label_names[Y[i]]} (num {Y[i]})")
            plt.pause(1)
        """

main()