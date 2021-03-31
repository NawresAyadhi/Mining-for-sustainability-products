# mining-for-product-sustainability 

1- You start by creating a conda environement, then installing all the dependencies: 

```bash
pip install -r requirements.txt
```

2- Then you scrape the data from **amazon.com** by running: 

```bash
scrapy runspider data_loading/amazon_reviews_scraping/spiders/sticks_amazon_reviews.py -o data/reviews.csv
```

3- To train a model on the data you can run the **sentiment_analysis.py**, this will output a model and save it to the **output/** folder

4- If you want to just test the already trained model you can just use the **inference.py** file as shown the **test.py** file
