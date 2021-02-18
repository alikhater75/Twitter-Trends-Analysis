# Twitter-Trends-Analysis

Here I try to analyze twitter top trends in UK. I built a model to create a 50-dimensional vector for each document. That is, documents with close content get close vectors.
Then I build a KNN model to predict the sentiment of every vector whether it's positive or negative. I used Twitter Api to retraive tweets about the top trends in UK. Finally I used the models I built to predict the sentiment for these tweets. 

# Running Project Locally
- Run `build_DocToVec_model.py` to create `DocToVec.d2v`
- Run `build_KNN_model.py` to create `KNN_model.joblib` or you can just use the pre_trained model (The attached file named `build_KNN_model.py`)
- Run `twitter_trends_analysis.py` to retraive tweets, analyze them and get some graphs to summarize the results.
- `KNN_model.joblib` is a pre_trained model.
- 'sentiment labelled sentences' file is essential to run (`build_KNN_model.py`).
- 'review_polarity' and 'aclImdb' are essential to run (`build_DocToVec_model.py`).
- 'twitter.jpg' picture is essential to run (`twitter_trends_analysis.py`).

# Resources
You can download "aclImdb" file from http://ai.stanford.edu/~amaas/data/sentiment/

You can download "review_polarity" from http://www.cs.cornell.edu/people/pabo/movie-review-data/

You can download "sentiment labelled sentences" from https://archive.ics.uci.edu/ml/machine-learning-databases/00331/