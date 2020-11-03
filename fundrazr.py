import scrapy


class items(scrapy.Item):
    username = scrapy.Field()
    follows_count = scrapy.Field()
    followed_by_count = scrapy.Field()
    is_verified = scrapy.Field()
    biography = scrapy.Field()
    external_link = scrapy.Field()
    full_name = scrapy.Field()
    posts_count = scrapy.Field()
    posts = scrapy.Field()

