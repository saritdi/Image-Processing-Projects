"""
Authors:
    1. Sarit Divkar     ID: 327373684
    2. Hadar Bar-Oz     ID: 204460737
"""

import os
import sys

import cv2
import numpy as np
import pandas as pd
import math
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from skimage import feature

"""
In pre-processing method at the first we will convert all the letters to a uniform size.
at the beginning its will covert image to grayscale and add padding to the image in order to get square size.
"""
def pre_processing(path):
    images = []
    label = []
    MY_DIRECTORY = os.listdir(path)
    # iterating over 26 folders in TEST or TRAIN
    for folder in MY_DIRECTORY:
        # iterating on images in each of the folders.
        path_folder = os.path.join(path, str(folder))
        #27
        if(int(folder)<27):
            for image in os.listdir(path_folder):
                path_image = os.path.join(path_folder, str(image))
                #input image from path
                img = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
                # padding
                height, width = img.shape
                if height>width:
                    add_padding = (height-width)/2
                    image_padded = cv2.copyMakeBorder(img, 0, 0, math.floor(add_padding), math.ceil(add_padding), cv2.BORDER_REPLICATE)
                else:
                    add_padding = (width-height)/2
                    image_padded = cv2.copyMakeBorder(img, math.floor(add_padding), math.ceil(add_padding), 0, 0, cv2.BORDER_REPLICATE)
                # resize
                resized_image = cv2.resize(image_padded, (40, 40))
                # add List of labels
                label.append(folder)
                # add to List of resized images
                images.append(resized_image)

    return [np.asarray(images), np.asarray(label)]

# HOG
def hog_images(images):
    hog_list = []
    for img in images:
        ch_hog = feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2), transform_sqrt=False, block_norm="L2")
        hog_list.append(ch_hog)
    return np.asarray(hog_list)

def chi_square(p, q):
  """
  Method to calculate chi square distance between two vectors. returns chi square distance
  """
  return 0.5 * np.sum((p-q)**2/(p+q+1e-6))

#timer
start = datetime.now()
print("Start:", start.time())
#takes the path of db from the cmd
cmd_path = sys.argv
script_dir = str(cmd_path[1])

path1 = script_dir + '\TRAIN'
path2 = script_dir + '\TEST'

#pre processing images
train_images, train_images_labels = pre_processing(path1)
test_images, test_images_labels = pre_processing(path2)

# HOG the processed images
train_hogged_images = np.array(list(map(lambda x: feature.hog(x, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=False, block_norm="L2") , train_images)))
test_hogged_images = np.array(list(map(lambda x: feature.hog(x, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=False, block_norm="L2") , test_images)))

# split train hog features to train and validation
train_hog_images_split, val_hog_images_split, train_labels_split, val_labels_split = train_test_split(train_hogged_images, train_images_labels, train_size=0.9, random_state=42)

# get the best k (between 1-15) and distance metric for knn
best_k = 1
best_accuracy_score = 0.0

# for each odd k between 1 to 15
for k in range(1, 16, 2):
    # create KNN model with euclidean distance
    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    # train model
    model.fit(train_hog_images_split, train_labels_split)
    # validate model
    value_prediction = model.predict(val_hog_images_split)
    # get accuracy
    accuracy_temp = accuracy_score(val_labels_split, value_prediction)
    # for now let's say euclidean is best
    best_distance_metric = 'euclidean'
    # if accuracy is better
    if accuracy_temp > best_accuracy_score:
        # update accuracy score
        best_accuracy = accuracy_temp
        # update k
        best_k = k

# for each odd k between 1 to 15
for k in range(1, 16, 2):
    # create KNN model with chi_square distance
    model = KNeighborsClassifier(n_neighbors=k, metric=chi_square)
    # train model
    model.fit(train_hog_images_split, train_labels_split)
    # validate model
    value_prediction = model.predict(val_hog_images_split)
    # get accuracy
    accuracy_temp = accuracy_score(val_labels_split, value_prediction)
    # if accuracy is better
    if accuracy_temp > best_accuracy_score:
        # update accuracy score
        best_accuracy = accuracy_temp
        # update distance metric
        best_distance_metric = 'chi square'
        # update k
        best_k = k

# test our model on the test data
if best_distance_metric == 'chi square':
    model = KNeighborsClassifier(n_neighbors=best_k, metric=chi_square)
else:
    model= KNeighborsClassifier(n_neighbors=best_k, metric=best_distance_metric)

model.fit(train_hogged_images, train_images_labels)
test_predict = model.predict(test_hogged_images)

# test results as a dictionary
test_classification_report_dict = classification_report(test_images_labels, test_predict, output_dict=True)

#open file to write results of KNN
f = open("results.txt", "w")
f.write("k= "+ str(best_k) + ", distance function is " + str(best_distance_metric) + " and test accuracy is " + str(test_classification_report_dict["accuracy"])+'\n')
f.write("Labels \t Accuracy \n")
for label in range(27):
    f.write(str(label))
    f.write("\t")
    f.write(str(test_classification_report_dict[str(label)]['precision']))
    f.write("\n")
#close file
f.close()

# confusion matrix
confusion_matrix = confusion_matrix(test_images_labels, test_predict, labels=[f'{i}' for i in range(0, 27)])
confusion_matrix_dataframe = pd.DataFrame(data=confusion_matrix, index=[i for i in range(0, 27)], columns=[i for i in range(0, 27)])
confusion_matrix_dataframe.to_csv("confusion_matrix.csv")

# end timer
end = datetime.now()
print('End: ', end.time())
print('Total time: ', end-start)





