# pip install streamlit fbprophet yfinance plotly
import csv
import numpy
import streamlit as st
from datetime import date, timedelta
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
def movingaverage(interval, window_size):
    window = numpy.ones(int(window_size)) / float(window_size)
    return numpy.convolve(interval, window, 'same')


START = (date.today() - timedelta(days=300)).strftime("%Y-%m-%d")
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Trading Deck')
stocks = []
comp = []
with open('stock_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        stocks.append(row[0])
        comp.append(row[1])

# stocks = [row[0] for row in reader]
# print(stocks)
# st.text(stocks)
selected_company = st.selectbox('Choose the Stock', comp)
ind = comp.index(selected_company)
selected_stock = stocks[ind]
n_days = st.slider('Days of prediction:', 1, 30)
period =n_days


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())


# Plot raw data
def plot_raw_data():
    # fig = go.Figure()
    fig = go.Figure(data=[go.Candlestick(x=data['Date'],
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    # y_av = movingaverage(data['Open'],200 )
    # st.plotly_chart(y_av)
    st.plotly_chart(fig)


plot_raw_data()
# st.plotly_chart(y_av)
# Predict forecast with Prophet.
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
volume = go.Figure(go.Indicator(
    mode="gauge+number",
    value=450,
    title={'text': "Speed"},
    domain={'x': [0, 1], 'y': [0, 1]}
))

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_days} days')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# st.write("Forecast components")
# fig2 = m.plot_components(forecast)
# st.write(fig2)

import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
from newspaper import Article
from newspaper import Config
from wordcloud import WordCloud, STOPWORDS

nltk.download('vader_lexicon')  # required for Sentiment Analysis
now = dt.date.today()
now = now.strftime('%m-%d-%Y')
yesterday = dt.date.today() - dt.timedelta(days=1)
yesterday = yesterday.strftime('%m-%d-%Y')

nltk.download('punkt')
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:78.0) Gecko/20100101 Firefox/78.0'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 100
company_name = selected_company
# news_df = 0
# As long as the company name is valid, not empty...
if company_name != '':
    print(f'Searching for and analyzing {company_name}, Please be patient, it might take a while...')

    # Extract News with Google News
    googlenews = GoogleNews(start=yesterday, end=now)
    googlenews.search(company_name)
    result = googlenews.result()
    # store the results
    df = pd.DataFrame(result)
    print(df)
try:
    list = []  # creating an empty list
    for i in df.index:
        dict = {}  # creating an empty dictionary to append an article in every single iteration
        article = Article(df['link'][i], config=config)  # providing the link
        try:
            article.download()  # downloading the article
            article.parse()  # parsing the article
            article.nlp()  # performing natural language processing (nlp)
        except:
            pass
        # storing results in our empty dictionary
        dict['Date'] = df['date'][i]
        dict['Media'] = df['media'][i]
        dict['Title'] = article.title
        dict['Article'] = article.text
        dict['Summary'] = article.summary
        dict['Key_words'] = article.keywords
        list.append(dict)
    check_empty = not any(list)
    # print(check_empty)
    if check_empty == False:
        news_df = pd.DataFrame(list)  # creating dataframe
        print(news_df)

except Exception as e:
    # exception handling
    print("exception occurred:" + str(e))
    print('Looks like, there is some error in retrieving the data, Please try again or try with a different ticker.')


# Sentiment Analysis
def percentage(part, whole):
    return 100 * float(part) / float(whole)


# Assigning Initial Values
positive = 0
negative = 0
neutral = 0
# Creating empty lists
news_list = []
neutral_list = []
negative_list = []
positive_list = []

# Iterating over the tweets in the dataframe
for news in news_df['Summary']:
    news_list.append(news)
    analyzer = SentimentIntensityAnalyzer().polarity_scores(news)
    neg = analyzer['neg']
    neu = analyzer['neu']
    pos = analyzer['pos']
    comp = analyzer['compound']

    if neg > pos:
        negative_list.append(news)  # appending the news that satisfies this condition
        negative += 1  # increasing the count by 1
    elif pos > neg:
        positive_list.append(news)  # appending the news that satisfies this condition
        positive += 1  # increasing the count by 1
    elif pos == neg:
        neutral_list.append(news)  # appending the news that satisfies this condition
        neutral += 1  # increasing the count by 1

positive = percentage(positive, len(news_df))  # percentage is the function defined above
negative = percentage(negative, len(news_df))
neutral = percentage(neutral, len(news_df))

# Converting lists to pandas dataframe
news_list = pd.DataFrame(news_list)
neutral_list = pd.DataFrame(neutral_list)
negative_list = pd.DataFrame(negative_list)
positive_list = pd.DataFrame(positive_list)
# using len(length) function for counting
print("Positive Sentiment:", '%.2f' % len(positive_list), end='\n')
print("Neutral Sentiment:", '%.2f' % len(neutral_list), end='\n')
print("Negative Sentiment:", '%.2f' % len(negative_list), end='\n')

# Define labels and sizes for the pie chart
labels = [
    'Positive [' + str(round(positive)) + '%]',
    'Neutral [' + str(round(neutral)) + '%]',
    'Negative [' + str(round(negative)) + '%]'
]
sizes = [positive, neutral, negative]
colors = ['yellowgreen', 'blue', 'red']

# Create a pie chart using Matplotlib
fig, ax = plt.subplots()
patches, texts, autotexts = ax.pie(
    sizes, colors=colors, startangle=90, autopct='%1.1f%%'
)
ax.legend(patches, labels, loc="best")
ax.set_title("Sentiment Analysis Result for stock= " + company_name)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.dataframe(news_df)
# Display the pie chart in Streamlit
st.pyplot(fig)

