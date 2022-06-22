# Crawling Data

Web Crawling adalah teknik pengumpulan data yang digunakan untuk mengindeks informasi pada halaman menggunakan URL (Uniform Resource Locator) dengan menyertakan API (Application Programming Interface) untuk melakukan penambangan dataset yang lebih besar. Data yang dapat kamu kumpulkan dapat berupa text, audio, video, dan gambar.

Salah satu library python yang digunakan untuk crawling data adalah scrapy. 

# Installasi Module Scrapy

Cara menginstall module scrapy melalui command prompt dengan menuliskan code sebagai berikut

```{tableofcontents}
pip install scrapy
```



### Membuat Code Program Scraping

sebelum melakukan crawling, disini kita mengambil data scraping terlebih dahulu. kita mengambil link dari data yang akan kita crawling lalu kumpulan link nanti akan dijadikan csv.

```python
import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):

        arrayData = ['https://pta.trunojoyo.ac.id/c_search/byfac/4/']
        for i in range(2, 461):
            inArray = 'https://pta.trunojoyo.ac.id/c_search/byfac/4/' + str(i)
            arrayData.append(inArray)
        for url in arrayData:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for i in range(1, 6):
            yield {'link': response.css('#content_journal > ul > li:nth-child(' + str(i) + ') > div:nth-child(3) > a::attr(href)').extract()
                   }
```

# Menjalankan File Spider

disini kita akan menjalankan file spider, akan tetapi kita harus masuk ke dalam direktori spider terlebih dahulu, caranya dengan menjalankan code berikut pada terminal.

```python
scrapy runspider nama-file.py
```

# menyimpan data ke file excel

untuk menyimpan hasil data yang telah kita scraping , kita dapat menggunakan perintah

```python
scrapy runspider nama-file.py -o nama-file-excel.xlsx
```

data yang telah kita scraping bisa disimpan dengan format lain.

# Membuat Code Program Crawling

kita menggunakan module scrapy dan pandas untuk membaca file csv hasil dari scraping yang kita lakukan tadi. pada crawling data ini, kita akan mengambil data judul, penulis, dosen pembimbing 1, dosen pembimbing 2, dan abstrak

```python
import scrapy
import pandas as pd


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):

        dataExcel = pd.read_xlsx('Datascrawl.xlsx')
        indexData = dataCSV.iloc[:, [0]].values
        arrayData = []
        for i in indexData:
            ambil = i[0]
            arrayData.append(ambil)
        for url in arrayData:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        yield {
            'judul': response.css('#content_journal > ul > li > div:nth-child(2) > a::text').extract(),
            'penulis': response.css('#content_journal > ul > li > div:nth-child(2) > div:nth-child(2) > span::text').extract(),
            'dosen1': response.css('#content_journal > ul > li > div:nth-child(2) > div:nth-child(3) > span::text').extract(),
            'dosen2': response.css('#content_journal > ul > li > div:nth-child(2) > div:nth-child(4) > span::text').extract(),
            'abstrak': response.css('#content_journal > ul > li > div:nth-child(4) > div:nth-child(2) > p::text').extract(),
        }

```

setelah menjalankan program tersebut, kita bisa menjalankan file spider dan menyimpan file hasil crawling dengan cara seperti diatas.