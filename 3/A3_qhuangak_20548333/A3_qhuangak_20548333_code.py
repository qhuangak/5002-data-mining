import numpy as np
from sklearn.cluster import KMeans
from keras.applications.resnet50 import ResNet50
#from keras.applications.vgg16 import VGG16
#from keras.applications.vgg16 import preprocess_input
from sklearn.decomposition import PCA
import cv2
import csv

#The first step
#import images by sorted sequence
images = np.ndarray((5011, 224, 224, 3), dtype=np.uint8)
#images=[]
for i in range(0, 5011):
    filename = str(i).zfill(5) + '.jpg'
    img1 = cv2.imread('./images/' + filename)     #read the images from 0 to 5010
    res1 = cv2.resize(img1, (224, 224))           # change the size of images to 224*224*3
    res1 = cv2.cvtColor(res1, cv2.COLOR_BGR2RGB)  # Convert it from BGR to RGB
    images[i]=res1                                #use images[] to store each resize image

print('resize finished')

#The second step
# use ResNet50 to extract the features
#images = preprocess_input(images)
#VGG16_model = VGG16(weights='imagenet', include_top=False)
ResNet50_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))  
feat = ResNet50_model.predict(images)

# reshape the picture from 4 dimensions to 2 dimensions
feat = feat.reshape(feat.shape[0], feat.shape[1] * feat.shape[2] * feat.shape[3])
print("extract feature over")

# use PCA to compress the data 
pca = PCA()
feat_pca = pca.fit_transform(feat)
print('PCA over')

#The third step
# train the K-means model with 18 clusters
estimator = KMeans(n_clusters=11)     # build the model
estimator.fit(feat)                   # use the featrue data to fit the model
# get the label of fitted images
label_pred = estimator.labels_  

#The last process
# get the number of predict clusters
cluster_num = max(label_pred) + 1

# store all the data to a dict
data_dict = dict()

# the key of each dict is 1,2,3...,cluster_num-1
# initialize data_dict
for i in range(cluster_num):
    data_dict[i] = []

# store all the labels to the corresponding cluster
for i in range(len(label_pred)):
    data_dict[label_pred[i]].append('\'' + str.zfill(str(i), 5) + '\'')

# define max length of the clusters
max_len = max([len(x) for x in data_dict.values()])

# polishing all the clusters to have the same number of labels
for i in range(cluster_num):
    for j in range(max_len - len(data_dict[i])):
        data_dict[i].append('')

# import the data_dict as a csv file
with open('ResNet50.csv', 'w', newline='') as f:
    csv_f = csv.writer(f)

    head = ['Cluster ' + str(i + 1) for i in range(cluster_num)]
    csv_f.writerow(head)

    for i in range(max_len):
        row = [data_dict[j][i] for j in range(cluster_num)]
        csv_f.writerow(row)

