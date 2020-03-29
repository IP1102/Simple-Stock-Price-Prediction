# Simple-Stock-Price-Prediction
Used to predict stock price based on historical data. 

This is a very simple machine learning project on the financial data. 
I have chosen Reliance Industries' last 5 years stock price data from Yahoo. The data can be found [here](https://in.finance.yahoo.com/quote/RELIANCE.NS?p=RELIANCE.NS&.tsrc=fin-srch).

# The aprroach
As a technical indicator, I've used 15 Days Moving Average on my dataset. I cleaned the data and calculated 15 Days Moving Average based on the Adjusted Close Price.
To prepare the training set, I divided the dataset in chunks of 60 (Timestamp = 60) such that the 61st data is the output of the previous 60 days record. Similarly, 2nd record is created from 2nd data to 61st data and the output is 62nd data and so on. In simpler words, each output is based on the previous day's Adjusted Closing price.
I prepared the data and changed the shape according to LSTM requirements. I fed my training set to the LSTM model and used adam as the optimizer. 

After training the data, I predicted the value of my test data (last 2 months stock prices), which I had placed in a separate csv file. I prepared my data similarly to my training set.

Lastly, I plotted my results to visualize my predictions. 


This is a very naive approach to predict daily stock prices, rather stock price trends and in NO WAY can predict actual market trends. 

But if you're bored and want to increase the efficiency of this code, you can try out by using different technical indicators such as RSI, MACD, etc. 
You can also try and visualize results by changing the timestamps. Ofcourse, all these are dependent on the specific stocks. Volatile stocks are hard to predict.
Try taking poorly volatile stocks. Please feel free to do a pull request anytime.

Hope you like it!

Thanks!
