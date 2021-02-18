import re
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from PIL import Image
from collections import Counter
from wordcloud import WordCloud
from wordcloud import STOPWORDS
from nltk.corpus import stopwords
import tweepy as tw
import matplotlib.pyplot as plt
from joblib import load

path = 'the path your file are located at'
path = 'D:/projects/faculty/AI/'
KNN_model = load(path + 'KNN_model.joblib')
DocToVec = Doc2Vec.load(path + "DocToVec.d2v")


def extract_words(sent):
    sent = sent.lower()
    sent = re.sub(r'<[^>]+>', ' ', sent) # strip html tags
    sent = re.sub(r'(\w)\'(\w)', '\1\2', sent) # remove apostrophes
    sent = re.sub(r'\W', ' ', sent) # remove punctuation
    sent = re.sub(r'\s+', ' ', sent) # remove repeated spaces
    sent = re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', sent) #remove any single letter
    sent = sent.strip()
    return sent.split()

 

# set authentication parameters,these values are unique for each twitter developer account
oauth = {"consumer_key": 'your consumer_key',
            "consumer_secret": 'your consumer_secret',
            "access_token": 'your access_token',
            "access_token_secret": 'your access_token_secret'}

# retraiving tweets 
def retraive_tweets (oauth, num_of_trends) : 
    """
    Searches for the top 50 trending hashtags in twitter, then chooses the top (num_of_trends) 
    trends with highest number of tweets(retweets excluded). For each trend retraive 
    the 500 most popular tweets and analize each tweet whether it is positive or negative review
    of this trend and store the results at trends_result DataFrame.
    

    Parameters:
    oauth - dictionary with keys as [consumer_key, consumer_secret, access_token,access_token_secret ]
    these values are unique for each twitter developer account 
    
    num_of_trends - number of trends to be analized 
    
    
    Returns:
    trends_result - DataFrame with trends' names as columns' lables and the columns values   
    are the predicted sentiment for each tweet
    
    
    words_used - dictionary of trend (string), all words used in trend's tweets (string) pairs
    """

    auth = tw.OAuthHandler(oauth["consumer_key"], oauth["consumer_secret"])
    auth.set_access_token(oauth["access_token"], oauth["access_token_secret"])
    api = tw.API(auth, wait_on_rate_limit=True)
    
    #retraive trending hashtags information in USA (23424977 corresponds to USA)
    tags = api.trends_place(23424975)
    
    # extract trends' names from tags
    trends = [ (tags[0]['trends'][i]['name'],tags[0]['trends'][i]['tweet_volume'])  
              for i in range(len(tags[0]['trends'])) if tags[0]['trends'][i]['tweet_volume'] != None]
    
    # sort the trends by the number of tweets and the choose the highest (num_of_trends)
    top_trends = sorted(trends, key=lambda tup: tup[1], reverse=True)[:num_of_trends]
    date_since = "2021-01-01"
    
    trends_result = {}
    words_used = {}
    for trend in top_trends :
        tweets = tw.Cursor(api.search,
                      q=trend[0] + " -filter:retweets",
                      lang="en",
                      since=date_since,
                      tweet_mode='extended',
                      result_type='popular ').items(500)
        
        tweetsVec = []
        words = []
        for tweet in tweets : 
            try : 
                # clean tweets and remove hashtags from it (if it has any)
                text = re.sub('\n', ' ', tweet.full_text.replace('https', '') )
                text = re.findall('(^.+?)#', text)
                tweet_words = extract_words(text[0])
            except:
                tweet_words = extract_words(tweet.full_text.replace('https', '')) # if tweet has no hahtags
                
            words.append(" ".join(tweet_words))    
            # build a vector for a tweet using pretraind model (DocToVec)
            tweet_vetor = DocToVec.infer_vector(tweet_words, steps=10)
            tweetsVec.append(tweet_vetor)    
        trends_result[trend[0]] = KNN_model.predict(tweetsVec) # analize each tweet whether it is positive or negative using pretraind model (KNN_model)
        words_used[trend[0]] = " ".join(words) 
        
    trends_result = pd.DataFrame(trends_result)   
    
    return trends_result, words_used
        
# making pie graphs
def make_pie_charts (results):
    """
    Make a Pie chart for each trend's values in results DataFrame. Assumes the DataFrame
    has trends' names as columns' lables and the columns values are the predicted 
    sentiment for each tweet in this trend

    Parameters:
    results - pandas DataFrame 

    """
    fig, plots = plt.subplots(1, len(results.columns))
    fig.suptitle("People's review of ", fontsize=20, y=0.8)
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.set_size_inches(15, 10, forward=True)    
    colors = {'Positive': 'C0',
           'Negative': 'C1'}
    for i, column in enumerate(results.columns): 
        
        index = results[column].value_counts().index
        if (index == [1, 0]).all() :
            labels = ['Positive','Negative']
        else :
            labels = ['Negative', 'Positive']
        plots[i].pie(results[column].value_counts()[index], autopct='%1.1f%%',
                     labels=labels, colors=[colors[key] for key in labels], 
                     explode = (0.05, 0.05))
        plots[i].set_title('"'+column+'" trend', fontsize=15)
        plots[i].axis('off')

    fig.savefig('pie_charts.jpg', dpi=100)
    plt.show()        

# making wordcloud graphs
def make_wordcloud (text_dict) :
    """
    Make a wordcloud graph of the most used words in each trend. Assumes text_dict
    is a dictionary with trends' names as keys and all the words used in that trend's 
    tweets(same word may appear more than once) as values 

    Parameters:
    text_dict - dictionary of trend (string), all words used in trend's tweets (string) pairs

    """
    mask = np.array(Image.open(path + 'twitter.jpg'))
    fig, plots = plt.subplots(1, len(text_dict.items()))
    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.suptitle('Most common words in ', fontsize=30, y=0.8)
    fig.set_size_inches(25, 15, forward=True)
    for i, (name, text) in enumerate(text_dict.items()) :
        words_frequency = Counter(text.split()) # returns unique words and its frequency
        words_to_use = dict(words_frequency.most_common(400)) # choose the 400 most frequent words 
        stop_words = set(stopwords.words('english') + list(STOPWORDS) + ['co']) #words to be eliminated
        
        wordcloud = WordCloud( 
                        background_color ='white', 
                        stopwords = stop_words, 
                        min_font_size = 10,
                        mask = mask,
                        max_words = 500).generate_from_frequencies({a: b for a, b in words_to_use.items() if a not in stop_words})
        
        plots[i].imshow(wordcloud)
        plots[i].set_title('"'+name+'" trend', pad= 5, fontsize=25)
        plots[i].axis('off')
    fig.savefig('word_clouds.jpg', dpi=100)
    plt.show()
         

# Draw conclusion 
def Draw_conclusion(results, text_dict) : 
    make_pie_charts(results)
    make_wordcloud(text_dict)
    
def main():
    results, text_dict = retraive_tweets(oauth, 3)
    Draw_conclusion(results, text_dict)

if __name__ == "__main__":
    main()