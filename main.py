# pip install streamlit fbprophet yfinance plotly
import csv
import numpy
import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis")
data = ["Tesla earnings tanked by 3%", "I hate you"]
print(sentiment_pipeline(data))
# from st_openai_embeddings_connection import OpenAIEmbeddingsConnection
def movingaverage(interval, window_size):
    window= numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')
START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Trading Deck')
stocks=[]
comp=[]
with open('stock_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        stocks.append(row[0])
        comp.append(row[1])

# stocks = [row[0] for row in reader]
print(stocks)
st.text(stocks)
selected_company = st.selectbox('Choose the Stock',comp)
ind = comp.index(selected_company)
selected_stock= stocks[ind]
n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365


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
    mode = "gauge+number",
    value = 450,
    title = {'text': "Speed"},
    domain = {'x': [0, 1], 'y': [0, 1]}
))

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

# st.write("Forecast components")
# fig2 = m.plot_components(forecast)
# st.write(fig2)
import plotly.graph_objects as go


fig = go.Figure(go.Bar(
            x=[-20, -14, 23],
            y=['giraffes', 'orangutans', 'monkeys'],
            orientation='h'))
st.plotly_chart(fig)
