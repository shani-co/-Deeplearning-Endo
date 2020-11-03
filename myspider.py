#pip install scrapy
# cat > myspider.py <<EOF
import scrapy
import csv

class BlogSpider(scrapy.Spider):
    name = 'blogspider'
    start_urls = ['https://www.stuffthatworks.health/endometriosis']

    def parse(self, response):
        for title in response.css('.post-header>h2'):
            yield {'title': title.css('a ::text').get()}

        for next_page in response.css('a.next-posts-link'):
            yield response.follow(next_page, self.parse)
# scrapy rSunspider myspider.py