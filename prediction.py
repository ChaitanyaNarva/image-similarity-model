def plot_(x,row,col,ind,title):

    """
    This function is used for plotting images and graphs (Visualization of end results of model training)
    Arguments:
    x - (np.ndarray or list) - an image array
    row - (int) - row number of subplot
    col - (int) - column number of subplot
    ind - (int) - index number of subplot
    title - (string) - title of the plot 
    """
    
    plt.subplot(row,col,ind)
    plt.imshow(x)
    plt.title(title)
    plt.axis('off')
    
def results_(query,result):

    """
    Plotting the N similar images from the dataset with query image.
    Arguments:
    query - (string) - filename of the query image
    result - (list) - filenames of similar images
    """
    
    def read(img):
        image = cv2.imread('/content/dataset/'+img)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return image
    plt.figure(figsize=(10,5))
    if type(query)!=type(30):
        plot_(query,1,1,1,"Query Image")
    else:
        plot_(read(files[query]),1,1,1,"Query Image "+files[query])
    plt.show()
    plt.figure(figsize=(20,5)) 
    for iter,i in enumerate(result):
        plot_(read(files[i]),1,len(result),iter+1,files[i])
    plt.show()
    
def predictions(label,N=8,isurl=False):

    """
    Making predictions for the query images and returns N similar images from the dataset.
    We can either pass filename or the url for the image.
    Arguments:
    label - (string) - file name of the query image.
    N - (int) - Number of images to be returned
    isurl - (string) - if query image is from google is set to True else False(By default = False)
    """
    
    start_time = time.time()
    if isurl:
        img = io.imread(label)
        img = cv2.resize(img,(224,224))
    else: 
        img_path = '/content/dataset/'+label
        img = image.load_img(img_path, target_size=(224,224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data,axis=0)
    img_data = preprocess_input(img_data)
    feature = K.function([model.layers[0].input],[model.layers[12].output])
    feature = np.array(feature).flatten().reshape(1,-1)
    res = knn.kneighbors(feature.reshape(1,-1),return_distance=True,n_neighbors=N)
    results_(img,list(res[1][0])[1:])
    print("Time taken : ",np.round(time.time()-start_time,2)," sec")
    
files = np.load('/content/drive/My Drive/files.npy')
optimizer = Adam(learning_rate=0.001)
knn = joblib.load('/content/drive/My Drive/knn_model.pkl')
kmeans = joblib.load('/content/drive/My Drive/kmeans_model.pkl')
model = load_model('/content/drive/My Drive/encoder_model.h5') 
model.compile(optimizer=optimizer, loss='mse')

query_path = '3057.jpg'
predictions(query_path)
