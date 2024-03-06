import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from pyexpat import model


# title and header
st.title("crypto price prediction")
st.write("This is a simple app to predict the price of crypto currency")

# sidebar
st.sidebar.header('User Input Parameters')  
genre = st.sidebar.radio('pic a currency to predict', ('BTC', 'ETC', 'XRP')) #radio button
day = st.sidebar.slider('Days of prediction', 0, 30, 1) #slider


#day att checker
if day == 1:
    att = "st"
elif day == 2:
    att = "nd"
elif day == 3:
    att = "rd"
else:
    att = "th"   


st.header(genre + ' price prediction')
# data set auto input
st.write('sample data set')
crypto_currency = genre
against_currency = 'USD'

start = dt.datetime(2021,1,1) #start date
end = dt.datetime.now() #end date
data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', start, end) #data reader
###st.write(data.tail()) #data tail

# original graph
st.write('original graph')
st.line_chart(data['Close'])

st.write("we are preperaing for the prediction of  ", day, att, 'day')

############------------------predection chart---------------------##############

#prepare Data
print(data.head())
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

pridiction_days = 60
#future_day = 30

x_train, y_train = [], []   

for x in range(pridiction_days, len(scaled_data)):   #scaled_data-future_day
    x_train.append(scaled_data[x-pridiction_days:x, 0])
    y_train.append(scaled_data[x, 0])   #x+future_day

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


#Build the Neural Network model

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) #Prediction of the next closing value

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

#model accuracy
accuracy = model.evaluate(x_train, y_train)
print(accuracy)

#Testing the model accuracy on existing data
test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data = web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', test_start, test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - pridiction_days:].values
model_inputs = model_inputs.reshape(-1,1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []
 
for x in range(pridiction_days, len(model_inputs)):
    x_test.append(model_inputs[x-pridiction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

#Predicting the next day    
real_data = [model_inputs[len(model_inputs) + day - pridiction_days:len(model_inputs+day), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))


prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)

st.write("prediction of the", day, att, "day is",prediction)#prints the prediction of the selected day

#plot the prediction
col1, col2 = st.columns(2)

with col1:
   st.header('actual prices')
   st.line_chart(actual_prices)

with col2:
   st.header("predicted prices")
   st.line_chart(predicted_prices)



