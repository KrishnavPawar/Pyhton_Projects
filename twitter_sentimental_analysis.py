import snscrape.modules.twitter as sntwitter
import re
import numpy as np
from textblob import TextBlob
import pandas as pd
from matplotlib import pyplot as plt

# Importing the tweets from twitter into a DataFrame
tweets = []
limit = 100
query = '(from:BillGates) since:2022-01-01'
for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    # print(vars(tweet))
    # break
    #print(tweet)
    if len(tweets) <=  limit:
        tweets.append([tweet.content,])
    else:
        break
df = pd.DataFrame(tweets, columns = ['content'])
print(df)
df.to_csv('raw_twitter_data.csv',index = False)

# Defining a fuction to clean the text
def cleantext(text):
    text = re.sub(pattern = '@[a-zA-Z0-9_]*',repl = ' ',string = text)     # Remobing retweets
    text = re.sub(pattern = '#',repl = ' ',string = text)                   # Removing Hastags
    text = re.sub(pattern = 'RT',repl = '',string = text)                  # Removing retweets
    text = re.sub(pattern = 'https?:\/\/\S+',repl = '',string = text)      # Removing hyper link  â€™
    text = re.sub(pattern = 'â€™',repl = "",string = text)
    return text
# Calling the function to clean the text
df['content'] = df['content'].apply(cleantext)
df['content']

# Creating function to  get the subjectivity
def getsubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# Creating function to get the polarity
def getpolarity(text):
    return TextBlob(text).sentiment.polarity

#creating 2 columns and printing subjectivity and polarity in that
df['polarity'] = df['content'].apply(getpolarity)
df['subjectivity'] = df['content'].apply(getsubjectivity)

# Saving the final data
# df.to_csv('final_twitter_data.csv')

# Creating a function to compute the negative, positive and neutral analysis
def getanalysis(score):
    if score > 0:
        return 'Positive'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Negative'
df['Analysis'] = df['polarity'].apply(getanalysis)

# printing out the most positive tweets
def printptweets():
    pcount = 1
    sorteddf = df.sort_values(by = ['polarity'],ascending = False)
    for i in range(0,sorteddf.shape[0]):
        if sorteddf['Analysis'][i] == 'Positive':
            print(str(pcount)+') '+sorteddf['content'][i])
            pcount += 1

# For printing out the most negative comment
def printntweets():
    ncount = 1
    sorteddf = df.sort_values(by = ['polarity'],ascending = True)
    for i in range(0,sorteddf.shape[0]):
        if sorteddf['Analysis'][i] == 'Negative':
            print(str(ncount)+') '+sorteddf['content'][i])
            ncount += 1

# plotting the polarity and subjectivity on scatter plot
def printscatterplot():
    plt.scatter(x = df['polarity'], y = df['subjectivity'])
    plt.xlabel("Polarity")
    plt.ylabel('Subjectivity')
    plt.title("Polarity vs Subjectivity")
    plt.grid()
    plt.show()

# Getting the percentage of positive tweets
ptweets = df[df.Analysis == 'Positive']
ptweets = ptweets['content']
print('percentage of positive tweets = ', round(ptweets.shape[0]/df.shape[0]*100,1) ,"%")

# Getting the percentage of negative tweets
ptweets = df[df.Analysis == 'Negative']
ptweets = ptweets['content']
print('percentage of negative tweets = ' , round(ptweets.shape[0]/df.shape[0]*100,1) ,"%")

# Plotting final Sentiment analysis in pie chart
def printpieplot():
    plt.figure(figsize = (5,5))
    plt.pie(x= df['Analysis'].value_counts(),explode=(0.04,0.04,0.04),autopct='%1.1f%%',labels=['Positive ','Neutral','Negative'],shadow = True)
    plt.title('Sentiment Analysis')
    plt.show()

# Saving the final data
df.to_csv('final_twitter_data.csv')


# TextBlob is a Python (2 and 3) library for processing textual data.
# It provides a simple API for diving into common natural language processing (NLP) tasks such as part-of-speech tagging,
# noun phrase extraction, sentiment analysis, classification, translation, and more.
