# ML Project - Music-recommendation-system

This project was created as a part of Machine Learning Course(CSE343) at IIIT Delhi.

### Group Member
 [Jasdeep Singh](https://github.com/Jassi-71 "GitHub Profile")
 [Siddharth Singh Kiryal](https://github.com/siddharth23ux "GitHub Profile")
 [Shanu Verma](https://github.com/Shanu85 "GitHub Profile")

## Introduction
In the present day scenario, most people listen to music. There are 79 million songs present on the internet. The quantity of songs has exploded due to the increased popularity of music streaming services. It has grown laborious and dif icult for consumers to get hold of similar songs of their preference. It includes individuals listening to and classifying various tunes based on their acoustic capabilities. Moreover, the recommendation system based on the song's metadata may not give them the best user experience. We wish to provide them with a better music experience by audio sampling the wav files and extracting the features. The goal is to improve efficiency and introduce automation for the said task. Machine learning techniques of today and Visualization tools should aid in discovering accurate models which recommend songs for users on music selections, as this was not addressed in previous research.

## DataSet and Evaluation
Obtained 1000 files from the GTZAN dataset [a link](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification).
Added two genres with 100 wav files each (Electric and k-pop). Extracted 30 sec from each wav file (from 30s offset) for further audio signal processing. Extracted features include RMS, chroma_stft, spectral_centroid, and many others using Librosa to create a 1200x43 dimensional dataset. For testing ANN, KNN, and logistic performance, the dataset was divided into 720:480.

The spectrogram and wavelet plots of all the 30s audio files were generated, shuffled, and divided into train and test datasets for image-based classification purposes. 
Spectrogram plots were used as inputs for image-based genre classification using CNN modes. The images were rescaled and converted into 256 x 256 x 3 (RGB) format. Then all the pixel values for all the images were standardized using the formula: 

![image](https://user-images.githubusercontent.com/81826357/170856907-f4624929-ec0a-446b-a9f0-6f02f53ba267.png)

, where I is the image of a 3 dimensional array. This was followed by fitting several set image control parameters to the training dataset. All 12 genres were encoded (0-11) for coding and NN classification purposes. CNN model uses a 60:40 train-test split (720:480 samples).

In ANN, CNN, and KNN, the evaluation metrics use accuracy, precision, F1-score, and Recall score. 
Further insights into the model have been obtained by analyzing the confusion matrix, heat maps, and learning curves (for train-test accuracy and loss). Silhouette and Davies Bouldin Scores evaluate clusters' separation (purity) for clustering models.


## Methodology

### Content Recommendation approach

There are three main approaches to content recommendation. 

#### 1 Demographic Filtering

The first is Demographic Filtering, where recommendations are based not on the user preferences but on the overall ranking of the songs based on global reviews received by the songs. 
This technique is much better than randomly recommending songs and it is useful when you want to recommend content that is generally favored by the larger population of viewers. But of course, this approach does not at all factor in user preference. 

#### 2 Content-Based Filtering

The second approach is Content-Based Filtering. This technique uses the metadata about the song such as song title or artist name, track length, features, and genres of the songs to find other songs which are similar to the songs selected by the user. When the user selects a song, the song is matched against other songs in the database to find the songs that are the most similar to the selected song based on the above-mentioned parameters. The top matches are then presented to the user and this is how content-based filtering works in essence. The similarity metric being used can be something like cosine similarity, Pearson similarity, or any other such metric. This approach does take into account the user’s selection however it may fail to capture the general taste and inclination of the user. 

#### 3 Collaborative Filtering

The last approach is that of collaborative filtering. This approach does not require the song metadata to make recommendations, instead, it relies on the user ratings made by other users of the service. This approach tries to match the general taste and behavior of the active user to existing users in the database and then tries to predict what the active user might like. A major downfall of this approach, however, is that data sparsity can negatively impact its performance. The data about user ratings and reviews are likely to be sparse. Thus, used on its own, the technique is prone to poor performance.
Meta features of songs such as spectral bandwidth, RMSE value, zero-crossing rate, frequency, harmonic mean and many others are used to make the dataset.

As our objective is to make a recommendation system. For that, we require to have a collection of models that are best suited for the dataset. Hence, our initial step would be to explore all the models and find out which models are giving better performance in terms of a better fit for both training and testing sets. After that, we would use the selected models to make a recommendation system along with recommendations metrics.

For the recommendation system, we would be using content based filtering for the recommendation.


## Setup
To run this project, install it locally using npm:
```
$ cd Codes
$ jupyter nbconvert --execute CNN.ipynb
$ jupyter nbconvert --execute Clustering___Recommendation.ipynb
$ jupyter nbconvert --execute FeatureExtraction_EDA_KNN_Logistic_SVM_ANN.ipynb
$ jupyter nbconvert --execute Random_Forest.ipynb
```

## Code Examples
To extract 30 seconds from offset 30: `for %i in (*.mp3) do ffmpeg -ss 30 -t 30 -i "%i" "%~ni.wav`

## Plots showing observations
> Figure 1: ANN - Varying test and train accuracy with epochs <br />
![image](https://user-images.githubusercontent.com/81826357/170858268-6b368612-48c0-4edd-8d54-e8827e7d8951.png)

> Figure 2: ANN - Varying test loss and train loss with epochs <br />
![image](https://user-images.githubusercontent.com/81826357/170858266-ac19883e-c9a0-404b-9f00-f96b4aca9cb4.png)

> Figure 3: Silhouette_Score with number of clusters (K_Mean) <br />
![image](https://user-images.githubusercontent.com/81826357/170858263-c694b583-cc10-425a-b5ba-c1d37cf7ceed.png)

> Figure 4: Silhouette_Score with different PCA components in K_Mean(n_clusters=9 n_init=9 ) <br />
![image](https://user-images.githubusercontent.com/81826357/170858261-522b1414-48b6-42c4-bea5-14fdbea8538e.png)

> Figure 5: Accuracy varying with k number of neighbors <br />
![image](https://user-images.githubusercontent.com/81826357/170858258-5a68c299-dc6c-43fe-a5cf-1c3283035256.png)

> Figure 6: TSNE(n_components=2,n_iter=500) on original data <br />
![image](https://user-images.githubusercontent.com/81826357/170858253-156baaa2-452e-412b-910a-979b2e659059.png)

> Figure 7: Confusion matrix of ANN for predicted and true labels <br />
![image](https://user-images.githubusercontent.com/81826357/170858249-7f1934b1-108c-443b-b107-555e79aff67b.png)

> Figure 8: CNN model architecture used  <br />
![image](https://user-images.githubusercontent.com/81826357/170858246-60278427-86be-4de3-8290-ae6bea9bbb50.png)

> Figure 9: CNN - Varying validation and train accuracy with epochs (for analysis purpose) (is overfitted) <br />
![image](https://user-images.githubusercontent.com/81826357/170858241-54f7f43b-dadb-408e-9811-9097120e07ed.png)

> Figure 10: CNN - Varying validation and train loss with epochs (for analysis purpose) (is overfitted) <br />
![image](https://user-images.githubusercontent.com/81826357/170858237-e94c7df5-3e10-4c3b-936a-c7cb9ca7b4aa.png)

> Figure 11: Confusion matrix of CNN for predicted and true labels  <br />
![image](https://user-images.githubusercontent.com/81826357/170858187-ef265a79-835d-425e-93bc-45bae21e0489.png)

> Figure 13: Heat map of initial data set 1200 rows, 43 column <br />
![image](https://user-images.githubusercontent.com/81826357/170858190-ec4e1bf5-a9ad-45e3-8ad3-361e1911541a.png)

> Figure 12: Histogram for all the features extracted using librosa <br />
![image](https://user-images.githubusercontent.com/81826357/170858150-2a6e9e15-17c5-4126-a779-5865569665b7.png)

> Figure 14: Recommendation System for Ann using the classification model <br />
![image](https://user-images.githubusercontent.com/81826357/170858173-35eebea2-cfd1-40bd-8e06-b38b3ecec879.png)

> Figure 15: Recommendation System for KNN using K nearest neighbor on the best hyperparameter found on the classification model. <br />
![image](https://user-images.githubusercontent.com/81826357/170858138-ea7bc676-3e0b-4bc0-8e09-21e662aaab66.png)

> Figure 16: Best hyperparameters for SVM - KFold accuracy, classification report. <br />
![image](https://user-images.githubusercontent.com/81826357/170858129-0539c45b-7437-45e5-b3de-b02e798475c2.png)

> Figure 17 : Recommendation using Clustering Algorithms <br />
![image](https://user-images.githubusercontent.com/81826357/170858088-2029eebf-a2c8-4f1e-8a42-033651e72bcf.png)

> Figure 18: Validation Curve with basic Random Forest <br />
![image](https://user-images.githubusercontent.com/81826357/170858113-95d2623a-643b-48a4-b5ae-7ae90345c450.png)

> Figure 19: Validation curve with parameter tuned Random Forest <br />
![image](https://user-images.githubusercontent.com/81826357/170858106-416f307d-acb7-42b8-bd48-a3978556e64a.png)

> Figure 20: The model summary for ANN model <br />
![image](https://user-images.githubusercontent.com/81826357/170858100-d592aa48-b302-498a-a4e3-8406fcf8abad.png)
