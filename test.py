from inference import infer_sentiment, infer_sustainabilty

review=["Easy to connect to the TV and start using. I had no issues or problems connecting and still no issues after a month of use. ",
        " Saved me a lot of money!! Bought this, instead of a new tv & it works amazing!!! Does more & is easier than I thought it would be!!",
         "Thinking about buying another.", "Hd pictures ... easy and fast interface, is my second 4K roku" ]

infer_sentiment(review)
infer_sustainabilty(review)