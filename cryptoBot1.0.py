from binance.client import Client
from discord.ext import commands
from numpy.lib.function_base import _quantile_is_valid
import config
import asyncio
import pickle
import os
import matplotlib.pyplot as plt
import discord
import numpy as np
import pandas as pd
import threading
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from keras.models import load_model
from binance.enums import *
from textblob import TextBlob
import textblob
import tweepy
import re

#Log into binance api
client = Client(config.binanceApiKey, config.binanceApiSecret)

#Discord bot
bot = commands.Bot(command_prefix='!')

#Log into twitter
authHandler = tweepy.OAuthHandler(consumer_key=config.twitterApiKey, consumer_secret=config.twitterApiSecretKey)
authHandler.set_access_token(config.twitterAccessToken, config.twitterAccessTokenSecret)
api = tweepy.API(authHandler)

#Stock Object
class Stock:
    def __init__(self, symbol):
        self.symbol = symbol
        self.alreadyHave = False
        self.prices = []
        self.priceBoughtAt = 0
        self.quantityBought = 0
        self.marketClosed = False
        self.predictedPrices = []
        self.positiveTweets = 1
        self.negativeTweets = 0

#Create stocks with only symbols in place
stocks = []
tickers = client.get_all_tickers()

#Load model
model = load_model("model.h5")
prediction_days = 800
predictedPoints = 60

#Number of data points and refresh rate in seconds
dataPoints = 1000
refreshRate = 120
#Time to collect data = dataPoints * refreshRate / 60 mins

#Number of tweets to look at for sentiment analysis
tweetAmount = 100
twitterRefreshRate = 900

#Open file
try:
    with open("crypto.txt", "rb") as filehandler:
        stocks = pickle.load(filehandler)
        for stock in stocks:
            while len(stock.prices) > dataPoints:
                stock.prices.pop(0)

except:          
    with open("crypto.txt", "wb") as filehandler:
        pickle.dump(stocks, filehandler, pickle.HIGHEST_PROTOCOL)
    
if len(stocks) == 0:
    #Create stocks
    for ticker in tickers:
        stocks.append(Stock(ticker['symbol']))

"""
test = 0
for stock in stocks:
    if stock.marketClosed == False:
        print(stock.symbol)
        test = test + 1

print(test)
"""

async def collectData():
    try:
        #Loop forever
        while True:
            tickers = client.get_all_tickers()
            #Fill information till there are enough data points
            for stock in stocks:
                #stock.alreadyHave = False
                #stock.marketClosed = False
                for ticker in tickers:
                    if ticker['symbol'] == stock.symbol:
                        stock.prices.append(float(ticker['price']))
                        while len(stock.prices) > dataPoints:
                            stock.prices.pop(0)
                        break
            
            #Write information into file
            #os.remove("crypto.txt")
            with open("crypto.txt", "wb") as filehandler:
                pickle.dump(stocks, filehandler, pickle.HIGHEST_PROTOCOL)
            
            await asyncio.sleep(refreshRate)

    except Exception as e:
        print("Collect Data: " + str(e))

async def buy():
    try:
        while True:
            #print("working")
            generalChannel = bot.get_channel(805608327538278423)
            info = client.get_account()
            for stock in stocks:
                if len(stock.prices) >= dataPoints:                    
                    balances = info['balances']
                        
                    if stock.alreadyHave == False and stock.marketClosed == False:  
                        for balance in balances:
                            #Found currency to buy this stock
                            if float(balance['free']) > 0:
                                #print("working 2")
                                if stock.symbol.find(balance['asset']) != -1 and stock.symbol.find(balance['asset']) != 0:
                                    budget = float(balance['free']) * 0.5
                                    test = client.get_symbol_info(stock.symbol)
                                    try:
                                        minQuantity = float(test['filters'][2]['minQty'])
                                        maxQuantity = float(test['filters'][2]['maxQty'])
                                        stepSize = float(test['filters'][2]['stepSize'])
                                        minNotional = float(test['filters'][3]['minNotional'])
                                        minNotional = minNotional / stock.prices[-1]
                                        
                                    except Exception as e:
                                        print("Get Symbol Info: " + str(e))
                                        minNotional = -1
                                        stepSize = -1
                                        minQuantity = -1
                                        maxQuantity = -1
                                        stock.marketClosed = True
                                                      
                                    quantity = 0    
                                    while minNotional > quantity or ((quantity + stepSize) * stock.prices[-1] < budget and quantity + stepSize < maxQuantity):
                                        quantity = quantity + stepSize
                                        
                                    quantity = float(round(quantity, 6))

                                    #print(str(minQuantity) + " < " + str(quantity))
                                    #Implied but just incase
                                    if minNotional != -1 and quantity >= minQuantity and quantity <= maxQuantity and (quantity - minQuantity) % stepSize == 0:
                                        fees = client.get_trade_fee(symbol=stock.symbol)
                                        makerFees = float(fees['tradeFee'][0]['maker'])
                                        takerFees = float(fees['tradeFee'][0]['taker'])
                                        fees = makerFees + takerFees
                                        priceChange = (float(stock.predictedPrices[-1]) - stock.prices[-1])
                                        #percentChange = (priceChange / stock.prices[-1]) * 100
                                        moneyMade = priceChange * quantity - fees
                                        #print("MoneyMade: " + str(moneyMade))
                                        #print("Check: " + str(quantity * stock.prices[-1] * 0.02))
                                        #Buy
                                        if moneyMade > 0:
                                            print("Buy " + stock.symbol)
                                            orderSuccess = order(SIDE_BUY, quantity, stock.symbol)
                                            if orderSuccess == "success":
                                                print("Bought")
                                                #await generalChannel.send("Bought " + str(quantity) + " of " + str(stock.symbol))
                                                #await generalChannel.send("Total: " + str(budget))
                                                stock.priceBoughtAt = stock.prices[-1] 
                                                stock.quantityBought = quantity
                                                stock.alreadyHave = True

                                            elif orderSuccess.find("Market is closed"):
                                                stock.marketClosed = True                            
            
                await asyncio.sleep(1)   

    except Exception as e:
        print("Buy: " + str(e))

async def sell():
    try:
        while True:
            #print("working")
            generalChannel = bot.get_channel(805608327538278423)
            info = client.get_account()
            for stock in stocks:
                if len(stock.prices) >= dataPoints:                    
                    balances = info['balances']
                        
                    if stock.alreadyHave == True:
                        fees = client.get_trade_fee(symbol=stock.symbol)
                        makerFees = float(fees['tradeFee'][0]['maker'])
                        takerFees = float(fees['tradeFee'][0]['taker'])
                        fees = makerFees + takerFees
                        priceChange = (stock.prices[-1] - stock.priceBoughtAt)
                        # percentChange = (priceChange / stock.priceBoughtAt) * 100
                        moneyMade = priceChange * stock.quantityBought - fees

                        # highestPredicted = 0
                        # for price in stock.predictedPrices:
                        #     if highestPredicted < price:
                        #         highestPredicted = price  
                        
                        if abs(moneyMade) > stock.quantityBought * stock.priceBoughtAt * 0.02 or (moneyMade > 0 and stock.predictedPrices[-1] < stock.prices[-1]):
                            print("Sell " + stock.symbol)
                            for balance in balances:
                                if float(balance['free']) > 0 and stock.symbol.find(balance['asset']) == 0:
                                    stock.quantityBought = float(balance['free'])   
                                    break
                            
                            maxQuantity = stock.quantityBought
                            test = client.get_symbol_info(stock.symbol)
                            try:
                                #maxQuantity = float(test['filters'][2]['maxQty'])
                                stepSize = float(test['filters'][2]['stepSize'])
                                minNotional = float(test['filters'][3]['minNotional'])
                                minNotional = minNotional / stock.prices[-1]
                                
                            except Exception as e:
                                print("Get Symbol Info: " + str(e))
                                minNotional = -1
                                stepSize = -1
                                #maxQuantity = -1
                                stock.marketClosed = True
                                                
                            quantity = 0    
                            while minNotional > quantity or quantity + stepSize < maxQuantity:
                                quantity = quantity + stepSize
                                
                            quantity = float(round(quantity, 6))
                            
                            orderSuccess = order(SIDE_SELL, quantity, stock.symbol)
                            if orderSuccess == "success":
                                print("Sold")
                                #await generalChannel.send("Sold " + str(stock.quantityBought) + " of " + str(stock.symbol))
                                #await generalChannel.send("Money Made: " + str(moneyMade))
                                stock.priceBoughtAt = 0
                                stock.quantityBought = 0
                                stock.alreadyHave = False  

                            else:
                                quantity = quantity - stepSize
                                orderSuccess = order(SIDE_SELL, quantity, stock.symbol) 
                                if orderSuccess == "success":
                                    print("Sold finally")   
                                    stock.priceBoughtAt = 0
                                    stock.quantityBought = 0
                                    stock.alreadyHave = False  

                                else:
                                    stock.alreadyHave = False                      
        
                await asyncio.sleep(1)   

    except Exception as e:
        print("Sell: " + str(e))

async def predictPrice():
    try:
        while True:            
            for stock in stocks:
                if len(stock.prices) >= prediction_days:
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    prices = np.array(stock.prices).reshape(-1, 1)
                    scaler = scaler.fit(prices)
                    total_dataset = stock.prices

                    model_inputs = np.array(total_dataset[len(total_dataset) - prediction_days:]).reshape(-1, 1)
                    model_inputs = scaler.transform(model_inputs)

                    #Predict Next period
                    real_data = [model_inputs[len(model_inputs) - prediction_days:len(model_inputs + 1), 0]]
                    real_data = np.array(real_data)
                    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

                    prediction = model.predict(real_data)
                    prediction = scaler.inverse_transform(prediction)
                    
                    stock.predictedPrices.append(prediction)
                    #print(stock.symbol + ": " + str(prediction))
                    while len(stock.predictedPrices) > predictedPoints:
                        stock.predictedPrices.pop(0)                                

    except Exception as e:
        print("Predict Price: " + str(e))

async def train():
    try:
        while True:        
            for stock in stocks:
                if len(stock.prices) == dataPoints:
                    #Prepare data
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    #scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
                    prices = np.array(stock.prices)
                    scaled_data = scaler.fit_transform(prices.reshape(-1, 1))

                    x_train = []
                    y_train = []

                    for x in range(prediction_days, len(scaled_data) - predictedPoints):
                        x_train.append(scaled_data[x - prediction_days:x, 0])
                        y_train.append(scaled_data[x + predictedPoints, 0])

                    x_train, y_train = np.array(x_train), np.array(y_train)
                    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1)) 
                    
                    #Train
                    model.fit(x_train, y_train, epochs=100, batch_size=100)
                    
                    #Update model
                    model.save('model.h5')           

    except Exception as e:
        print("Train: " + str(e))

async def twitterReview():
    try:
        while True: 
            info = client.get_account()
            balances = info['balances']
            
            for balance in balances:
                coin = balance['asset']             
                search = f'#{coin} -filter:retweets'
                tweetCursor = tweepy.Cursor(api.search, q=search, lang='en', tweet_mode='extended').items(tweetAmount)
                tweets = [tweet.full_text for tweet in tweetCursor]
                tweetsDF = pd.DataFrame(tweets, columns=['Tweets'])

                for _, row in tweetsDF.iterrows():
                    row['Tweets'] = re.sub('http\S+', '', row['Tweets'])
                    row['Tweets'] = re.sub('#\S+', '', row['Tweets'])
                    row['Tweets'] = re.sub('@\S+', '', row['Tweets'])
                    row['Tweets'] = re.sub('\\n', '', row['Tweets'])

                tweetsDF['Polarity'] = tweetsDF['Tweets'].map(lambda tweet: textblob.TextBlob(tweet).sentiment.polarity)
                tweetsDF['Result'] = tweetsDF['Polarity'].map(lambda pol: '+' if pol > 0 else '-')

                positive = tweetsDF[tweetsDF.Result == '+'].count()['Tweets']
                negative = tweetsDF[tweetsDF.Result == '-'].count()['Tweets']

                for stock in stocks:
                    if stock.symbol.find(coin) == 0:
                        stock.positiveTweets = positive
                        stock.negativeTweets = negative
            
                await asyncio.sleep(twitterRefreshRate)

    except Exception as e:
        print("Twitter Review: " + str(e))

@bot.event
async def on_ready():
    try:
        t1 = threading.Thread(target=asyncio.run, args=(collectData(),))
        t1.start()
        t2 = threading.Thread(target=asyncio.run, args=(predictPrice(),))
        t2.start()
        t3 = threading.Thread(target=asyncio.run, args=(buy(),))
        t3.start()
        t4 = threading.Thread(target=asyncio.run, args=(sell(),))
        t4.start()
        #t5 = threading.Thread(target=asyncio.run, args=(twitterReview(),))
        #t5.start()
        t6 = threading.Thread(target=asyncio.run, args=(train(),))
        t6.start()

    except Exception as e:
        print("On Ready: " + str(e))      

@bot.command(name='time')
async def time(context):
    try:
        await context.message.channel.send(str(len(stocks[len(stocks) - 1].prices)) + "/" + str(dataPoints))

    except Exception as e:
        print("Time: " + str(e))

@bot.command(name='price')
async def price(context, arg1):
    try:
        tickers = client.get_all_tickers()
        price = None
        for stock in stocks:
            if stock.symbol == arg1:
                price = stock.prices[-1]  

                prices = []
                for i in range(len(stock.prices) - 200, len(stock.prices)):
                    prices.append(stock.prices[i])

                plt.style.use('dark_background')   
                plt.plot(prices, color='white', label=f"Actual {stock.symbol} Price")
                break
               
        if price == None:
            await context.message.channel.send("Not on binance")
        
        else:
            await context.message.channel.send(str(arg1) + ": " + str(price))
            plt.savefig(fname='plot')
            await context.message.channel.send(file=discord.File("plot.png"))
            os.remove("plot.png")
            plt.clf()

    except Exception as e:
        print("Price: " + str(e))

@bot.command(name='predict')
async def predict(context, arg1):
    try:      

        prediction = None
        for i in stocks:
            
            if i.symbol == arg1:
                prediction = i.predictedPrices[-1]
                predictedPrices = np.array(i.predictedPrices).reshape(-1)
                plt.style.use('dark_background')   
                plt.plot(predictedPrices, color='white', label=f"Predicted {i.symbol} Price")
                break
        
        if prediction == None:
            await context.message.channel.send("Isn't on binance")
        
        else:
            await context.message.channel.send("Prediction: " + str(prediction))
            plt.savefig(fname='plot')
            await context.message.channel.send(file=discord.File("plot.png"))
            os.remove("plot.png")
            plt.clf()
            
    except Exception as e:
        print("Predict: " + str(e))

def order(side, quantity, symbol):
    try:
        string = "success"
        order = client.create_order(symbol=symbol, side=side, type=ORDER_TYPE_MARKET, quantity=quantity)
    
    except Exception as e:
        print("Order: " + str(e))
        string = str(e)

    return string

@bot.command(name='status')
async def status(context):
    try:
        for i in stocks:
            currentPrice = i.prices[-1]

            if i.alreadyHave == True:
                priceChange = ((currentPrice - i.priceBoughtAt) / i.priceBoughtAt) * 100
                await context.message.channel.send(i.symbol + ': $' + str(i.priceBoughtAt) + " * " + str(i.quantityBought) + ' ------- priceChange = ' + str(round(priceChange, 2)))

    except Exception as e:
        print("Status: " + str(e))

#Run discord bot
bot.run(config.discordBot)
