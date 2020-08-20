# Image Similarity Model
# Finding N similar images from the dataset given on a query image.

# Plan of Action
![plan](https://user-images.githubusercontent.com/58396314/90765118-0b2d0f00-e307-11ea-9fbf-22e1823c99eb.PNG)

*    We have nearly ~5K images with 512x512 resolution gives ~1,310,720,000 pixels. Loading into RAM and processing each image with every other image will be computationally expensive and may crashes the system(GPU or TPU).
*    So as a solution I integrated both convolutional neural networks and autoencoder ideas for information reduction from image based data. 
*    That would be pre-processing step for clustering.

<img width="573" alt="conv_encoder" src="https://user-images.githubusercontent.com/58396314/90765423-7ecf1c00-e307-11ea-955d-ea059e6acb2f.PNG">

<strong>Convolutional AutoEncoders:</strong>
*    We can call left to centroid side as convolution whereas centroid to right side as deconvolution.
*    Deconvolution side is also known as unsampling or transpose convolution. It is a basic reduction operation.
*    Reverse operation using upsampling to decode the encoded image.
*    Building an autoencoder model, grabing the compressed image from the intermediate layers, then feed that lower-dimension array into KMeans clustering. 

<strong>K-Means Clustering</strong>
*    We can then apply clustering to compressed representation. I would like to apply k-means clustering to cluster the images into 4 groups.
*    This could fasten labeling process for unlabeled data.

<strong>KNN</strong>
*    Model training to find N similar images.
*    Finding Nearest neighbors and taking N nearest points as similar images given on a query image.

<strong>Prediction Algorithm:</strong>

   Step-1: taking either filename or url and converting that image into image array.
   
   Step-2: Using that array finding the feature from the intermediate layers of the trained autoencoder model.
   
   Step-3: From the extracted features finding the label to which that image belongs using K-Means clustering.
   
   Step-4: Using KNN model finding N similar images using predict images and finally plotting the result.
