from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")
data = ["Tesla earnings tanked by 3%", "I hate you"]
sentiment_pipeline(data)
import requests

url = ('https://newsapi.org/v2/top-headlines?country=in&category=business&apiKey=5f52da7203904a0cae3a2282b2b861af')

response = requests.get(url)
news=response.json()
articles=news.get('articles')
print(articles)
titles, descriptions = [], []
for article in articles:
    title = article.get('title')
    description = article.get('description')

    titles.append(title)
    descriptions.append(description)

print(titles, descriptions)
desc_sentiment=[]
for i in range(len(descriptions)):
    try:
        data= (descriptions[i])
        sentiment = sentiment_pipeline(data)
        print(sentiment)
        desc_sentiment.append(sentiment)
    except Exception as e:
        continue
title_sentiment=[]
for i in range(len(titles)):
    try:
        data= (titles[i])
        sentiment = sentiment_pipeline(data)
        print(sentiment)
        title_sentiment.append(sentiment)
    except Exception as e:
        continue
print(title_sentiment)
label,score=[],[]
for k in title_sentiment:
    l=k[0].get('label')
    label.append(l)
    s=k[0].get('score')
    score.append(s)

net = 0
for i in range(len(label)):

    if label[i] == 'NEGATIVE':
        net = net + (-1 * score[i])
    if label[i] == 'POSITIVE':
        net = net + (+1 * score[i])
label,score=[],[]
for k in desc_sentiment:
    l=k[0].get('label')
    label.append(l)
    s=k[0].get('score')
    score.append(s)
net = 0
for i in range(len(label)):

    if label[i] == 'NEGATIVE':
        net = net + (-1 * score[i])
    if label[i] == 'POSITIVE':
        net = net + (+1 * score[i])