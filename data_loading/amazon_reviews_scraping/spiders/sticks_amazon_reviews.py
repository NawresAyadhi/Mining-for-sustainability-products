import scrapy


class MyScraper(scrapy.Spider):
    name = 'MyScraper'
    allowed_domains = ['amazon.com']
    start_urls = ['http://amazon.com/']

    # Base URL for the MacBook air reviews
    Url_Tv_Stick = "https://www.amazon.com/fire-tv-stick-lite/product-reviews/B07YNLBS7R/ref=cm_cr_arp_d_viewopt_rvwer?ie=UTF8&reviewerType=avp_only_reviews&pageNumber="
    Url_Roku_Stick="https://www.amazon.com/Roku-Streaming-Stick-HDR-Streaming-Long-range/product-reviews/B075XLWML4/ref=cm_cr_arp_d_viewopt_rvwer?ie=UTF8&reviewerType=avp_only_reviews&pageNumber="
    start_urls=[]

    # Creating list of urls to be scraped by appending page number a the end of base url
    for i in range(1,900):
        start_urls.append(Url_Tv_Stick+str(i))
        start_urls.append(Url_Roku_Stick+str(i))

    # Defining a Scrapy parser
    def parse(self, response):
        data = response.css('#cm_cr-review_list')

        # Collecting product star ratings
        star_rating = data.css('.review-rating')
        #comment_title = data.css('.review-title')


        # Collecting user reviews
        comments = data.css('.review-text')
        count = 0

        # Combining the results
        for review in star_rating:
            yield{'sentiment': ''.join(review.xpath('.//text()').extract()),
                 'review': ''.join(comments[count].xpath(".//text()").extract()),
                 #'title': ''.join(comment_title[count].xpath(".//text()").extract())
                 }
            count=count+1