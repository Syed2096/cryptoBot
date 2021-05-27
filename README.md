# cryptoBot
CryptoBot for trading cyrptocurrenies.

How to Run:
1. Download neccesary libraries, python, etc.
2. Get twitter token or comment it out this feature wasn't very good, discord server token, binance tokens, etc.
3. Add tokens to config file.
4. Also change discord channel ID to your own channel ID. (lines 221 and 309)
5. Start the program. 
8. Use !price "exchange" and !predict "exchange", to view the bots thought process. "exchange" refers to BTCUSDT, DOGEBTC and all other coins on binance.
9. !price will start showing predictions on the same graph as prices once there are enough predictions.
10. Once, you are happy with the bots predictions stop the program. (Recommended time to run before next step would be 2 days)
11. Uncomment out lines 165 - 168.
12. Delete crypto.txt and restart program.

Warning: Your entire binance account will be used by the bot, you could lose everything. You can still make trades yourself, however the bot will make any trades it sees fit.
This bot gets better the longer it runs, eventually it should theoretically be able to make a profit more often than not.

I am currently in the process of testing the bot out myself, so far its been good but it requires a lot of time to train the model.
