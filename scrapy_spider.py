import scrapy
from bs4 import BeautifulSoup
from PIL import Image


class MySpider(scrapy.Spider):
    name = 'my_spider'
    results = []  # Add this line

    def start_requests(self):
        PZNs = self.settings.get('PZNs')
        shop = self.settings.get('shop')

        for PZN in PZNs:
            if shop == 'mycare.de':
                url = f"https://www.mycare.de/suche/dewd?q={PZN}"
            else:
                url = f"https://www.shop-apotheke.com/arzneimittel/{PZN}/"

            yield scrapy.Request(url, callback=self.parse, meta={'PZN': PZN})

    def parse(self, response):
        PZN = response.meta['PZN']
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract and process the images
        images = soup.find_all('img')
        extracted_images = []

        for image in images:
            src = image.get('src')

            # Process the image (resize, save, etc.)
            # ...
            # Example: Resizing the image to 256x256 pixels
            img = Image.open(requests.get(src, stream=True).raw)

            # Save the processed image to a file
            img_filename = f"static/images/{PZN}_{len(extracted_images)}.jpg"
            img.save(img_filename)

            extracted_images.append({
                'url': src,
                'filename': img_filename,
            })

        self.results.append({
            'PZN': PZN,
            'results': extracted_images,
        })
