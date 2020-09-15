# Image Similarity Model
# Finding N similar images from the dataset given on a query image.

# Plan of Action
![plan](https://user-images.githubusercontent.com/58396314/90765118-0b2d0f00-e307-11ea-9fbf-22e1823c99eb.PNG)

*    We have nearly ~5K images with 512x512 resolution gives ~1,310,720,000 pixels. Loading into RAM and processing each image with every other image will be computationally expensive and may crashes the system(GPU or TPU).
*    So as a solution I integrated both convolutional neural networks and autoencoder ideas for information reduction from image based data. 
*    That would be pre-processing step for clustering.


<strong>Convolutional AutoEncoders:</strong>
*    We can call left to centroid side as convolution whereas centroid to right side as deconvolution.
*    Deconvolution side is also known as unsampling or transpose convolution. It is a basic reduction operation.
*    Reverse operation using upsampling to decode the encoded image.
*    Building an autoencoder model, grabing the compressed image from the intermediate layers, then feed that lower-dimension array into KMeans clustering. 

![convo;utional_autoencoders](https://user-images.githubusercontent.com/58396314/93178450-e5eabf80-f751-11ea-9bd6-e4714bab7161.PNG)

<strong>K-Means Clustering</strong>
*    We can then apply clustering to compressed representation. I would like to apply k-means clustering to cluster the images into 4 groups.
*    This could fasten labeling process for unlabeled data.

![clustering](https://user-images.githubusercontent.com/58396314/93178167-78d72a00-f751-11ea-8a55-9fb06ad83589.PNG)

<strong>Dimensionality Reduction Through T-SNE:</strong>
To visualize the clustering we need to perform dimensionality reduction through T-SNE. Which helps us to decide the optimal hyperparameter.

<strong>K-Nearest Neighbors(KNN)</strong>
*    Model training to find N similar images.
*    Finding Nearest neighbors and taking N nearest points as similar images given on a query image.

![knn](https://user-images.githubusercontent.com/58396314/93177880-14b46600-f751-11ea-8311-112320b0f1af.PNG)

<strong>Prediction Algorithm:</strong>

   Step-1: taking either filename or url and converting that image into image array.
   
   Step-2: Using that array finding the feature from the intermediate layers of the trained autoencoder model.
   
   Step-3: From the extracted features finding the label to which that image belongs using K-Means clustering.
   
   Step-4: Using KNN model finding N similar images using predict images and finally plotting the result.

<strong>Real Time Testing1:</strong>

![query](https://user-images.githubusercontent.com/58396314/93178996-b25c6500-f752-11ea-901e-8cc0ce4734e2.PNG)

<strong>Real Time Testing2:</strong>
![testing3](https://user-images.githubusercontent.com/58396314/93179232-0109ff00-f753-11ea-9f30-e8b0c30695c0.PNG)

