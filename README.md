# cryptoBot
CryptoBot for trading cyrptocurrenies.

How to Run:
1. Download neccesary libraries, python, etc.
2. Get twitter token or comment it out this feature wasn't very good, discord server token, binance tokens, etc.
3. Add tokens to config file.
4. Also change discord channel ID to your own channel ID. (lines 180, 220 and 308)
5. Start the program. 
8. Use !price "exchange" and !predict "exchange", to view the bots thought process. "exchange" refers to BTCUSDT, DOGEBTC and all other coins on binance.
9. !price will start showing predictions on the same graph as prices once there are enough predictions.
10. Once, you are happy with the bots predictions stop the program. (Recommended time to run before next step would be 1 week, but no idea if the bot is even good)
11. Uncomment out lines 160 - 163, this will allow the bot to buy and sell. Also comment out lines 166 and 167, to stop messing with the bots brain.
12. Delete crypto.txt and start program.
13. !status will show current trades and price change. It will show nothing if no trades were made.

Warning: Your entire binance account will be used by the bot, you could lose everything. You can still make trades yourself, however the bot will make any trades it sees fit.
