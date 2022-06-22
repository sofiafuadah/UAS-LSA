# Text Preprocessing

Text preprocessing adalah suatu proses untuk menyeleksi data teks agar menjadi lebih terstruktur  lagi dengan melalui serangkaian tahapan yang meliputi tahapan case folding, tokenizing, filtering dan stemming.

# Import Library

sebelum memulai tahapan preprocessing teks, kita harus menuliskan library yang akan kita gunakan.  Pada code dibawah ini saya menggunakan library pandas, numpy, string, regex, NLTK dan SkLearn.

````{tableofcontents}
import pandas as pd
import numpy as np
#Import Library untuk Tokenisasi
import string 
import re #regex library

# import word_tokenize & FreqDist dari NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
````

# Read Data

sebelum masuk ke proses, kita harus membaca data hasil crawling yang telah kita lakukan tadi. pada code ini saya menggunakan library pandas untuk membaca file datanya.

```
dataPTA = pd.read_excel('PTAscrawl.xlsx')
```

# Case Folding

Case folding merupakan tahapan dalam preprocessing teks yang dilakukan untuk menyeragamkan karakter. Case folding merupakan istilah yang digunakan untuk mengubah semua bentuk huruf dalam sebuah teks atau dokumen menjadi huruf kecil semua. 

contoh case folding :

kata " Kantor Badan Kepegawaian" menjadi "kantor badan kepegawaian"

```
# gunakan fungsi Series.str.lower() pada Pandas
dataPTA['Abstrak'] = dataPTA['Abstrak'].str.lower()

print('Case Folding Result : \n')

#cek hasil case folding
print(dataPTA['Abstrak'].head(5))
print('\n\n\n')
```

# Tokenizing

Tokenizing adalah proses untuk membagi teks yang dapat kalimat, paragraf atau dokumen, menjadi token-token atau bagian tertentu.

contoh tokenizing :

" sumber daya alam" menjadi "sumber", "daya", "alam"

# Stopword Removal

stopword removal adalah proses untuk menghilangkan karakter yang tidak penting. Cotntohnya disini menghilangkan tanda baca, nomor.

```
def remove_PTA_special(text):
    # menghapus tab, new line, dan back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # menghapus non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # menghapus mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # menghapus incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
                
dataPTA['Abstrak'] = dataPTA['Abstrak'].apply(remove_PTA_special)

#menghapus nomor
def remove_number(text):
    return  re.sub(r"\d+", "", text)

dataPTA['Abstrak'] = dataPTA['Abstrak'].apply(remove_number)

#menghapus punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

dataPTA['Abstrak'] = dataPTA['Abstrak'].apply(remove_punctuation)

#menghapus spasi leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

dataPTA['Abstrak'] = dataPTA['Abstrak'].apply(remove_whitespace_LT)

#menghapus spasi tunggal dan ganda
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

dataPTA['Abstrak'] = dataPTA['Abstrak'].apply(remove_whitespace_multiple)

# menghapus kata 1 abjad
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

dataPTA['Abstrak'] = dataPTA['Abstrak'].apply(remove_singl_char)

print('Result : \n') 
print(dataPTA['Abstrak'].head())
print('\n\n\n')
```

setelah itu dilakukan stopword bahasa Indonesia menggunakan library NLTK, dan juga extend kata-kata yang tidak memiliki makna

```
list_stopwords = stopwords.words('indonesian')
list_stopwords.extend(["aam","absolute","abstract","abstrakxd","adm","ahp","ai","aid","akanxd","akhirxd","alert","algorithm"])
# Mengubah List ke dictionary
list_stopwords = set(list_stopwords)
```

# Stemming

Stemming adalah proses pemetaan dan penguraian bentuk dari suatu kata menjadi bentuk kata dasarnya. gampangnya, stemming merupakan proses perubahan kata berimbuhan menjadi kata dasar.

contohnya : menginginkan menjadi ingin

