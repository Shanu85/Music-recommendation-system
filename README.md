# ML Project - Music-recommendation-system

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

