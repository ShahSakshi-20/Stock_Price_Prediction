#Importing
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data 
import yfinance as yf
from keras.models import load_model
# import streamlit as st

#Defining start and end date
start = '2012-01-01'
end = '2023-11-10'
stock_symbol = 'AAPL'

st.title('Stock Trend Predictior')
user_input = st.text_input("Enter Stock Ticker", 'AAPL')
#stock name for eg:TSLA, further this will become dynamic
df = yf.download(user_input, start=start, end=end)
# df = data.DataReader('APPL','yahoo',start,end)



#Describing the Data
st.subheader('Data from 2012 - 2023')
st.write(df.describe())

#Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)



#For 100MA
st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100 , 'r')
plt.plot(ma200 , 'g')
plt.plot(df.Close)
st.pyplot(fig)


#Splitting the data into training and testing

#  70% data is Training Data
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])

#  30% Data is Testing Data
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(data_training.shape)
print(data_testing.shape)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)



model = load_model('keras_model.h5')


#Testing Part

past_100_days = data_training.tail(100)

final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100 , input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#Final Graph

st.subheader('Predictions vs Originals')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)