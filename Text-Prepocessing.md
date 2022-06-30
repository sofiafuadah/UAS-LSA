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
dataPTA = pd.read_excel('hasilmanajemen.xlsx')
```

# Case Folding

Case folding merupakan tahapan dalam preprocessing teks yang dilakukan untuk menyeragamkan karakter. Case folding merupakan istilah yang digunakan untuk mengubah semua bentuk huruf dalam sebuah teks atau dokumen menjadi huruf kecil semua. 

contoh case folding :

kata " Kantor Badan Kepegawaian" menjadi "kantor badan kepegawaian"

```
# gunakan fungsi Series.str.lower() pada Pandas
dataPTA['abstrak'] = dataPTA['abstrak'].str.lower()

print('Case Folding Result : \n')

#cek hasil case fold
print(dataPTA['abstrak'].head(5))
print('\n\n\n')
```

# Tokenizing

Tokenizing adalah proses untuk membagi teks yang dapat kalimat, paragraf atau dokumen, menjadi token-token atau bagian tertentu.

contoh tokenizing :

" sumber daya alam" menjadi "sumber", "daya", "alam"

```

```



# Stopword Removal

stopword removal adalah proses untuk menghilangkan karakter yang tidak penting. Cotntohnya disini menghilangkan tanda baca, nomor.

```python
#Import Library untuk Tokenisasi
import string 
import re #regex library
dataPTA.abstrak = dataPTA.abstrak.astype(str)

# import word_tokenize & FreqDist dari NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

def remove_PTA_special(text):
    # menghapus tab, new line, dan back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # menghapus non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # menghapus mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # menghapus incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
                
dataPTA['abstrak'] = dataPTA['abstrak'].apply(remove_PTA_special)

#menghapus nomor
def remove_number(text):
    return  re.sub(r"\d+", "", text)

dataPTA['abstrak'] = dataPTA['abstrak'].apply(remove_number)

#menghapus punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

dataPTA['abstrak'] = dataPTA['abstrak'].apply(remove_punctuation)

#menghapus spasi leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

dataPTA['abstrak'] = dataPTA['abstrak'].apply(remove_whitespace_LT)

#menghapus spasi tunggal dan ganda
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

dataPTA['abstrak'] = dataPTA['abstrak'].apply(remove_whitespace_multiple)

# menghapus kata 1 abjad
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

dataPTA['abstrak'] = dataPTA['abstrak'].apply(remove_singl_char)

# Tokenisasi
def word_tokenize_wrapper(text):
    return word_tokenize(text)

dataPTA['abstrak_token'] = dataPTA['abstrak'].apply(word_tokenize_wrapper)

print('Tokenizing Result : \n') 
print(dataPTA['abstrak_token'].head())
```

setelah itu dilakukan stopword bahasa Indonesia menggunakan library NLTK, dan juga extend kata-kata yang tidak memiliki makna

```python
from nltk.corpus import stopwords

list_stopwords = stopwords.words('indonesian')

# Mengubah List ke dictionary
list_stopwords = set(list_stopwords)

#remove stopword pada list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

#aStopwording
dataPTA['abstrak_stop'] = dataPTA['abstrak_token'].apply(stopwords_removal) 

print(dataPTA['abstrak_stop'].head(20))
```

# Stemming

Stemming adalah proses pemetaan dan penguraian bentuk dari suatu kata menjadi bentuk kata dasarnya. gampangnya, stemming merupakan proses perubahan kata berimbuhan menjadi kata dasar.

contohnya : menginginkan menjadi ingin

```python
# import Sastrawi package
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter


# membuat stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# sfungsi stemmer
def stemmed_wrapper(term):
    return stemmer.stem(term)

term_dict = {}

for document in dataPTA['abstrak_stop']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
            
print(len(term_dict))
print("------------------------")

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    print(term,":" ,term_dict[term])
    
print(term_dict)
print("------------------------")


# stemming pada dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]

dataPTA['abstrak_stem'] = dataPTA['abstrak_stop'].swifter.apply(get_stemmed_term)
print(dataPTA['abstrak_stem'])
```

# Menghitung Term-document Matrix

sebelum meghitung term-document matrix (TF-IDF) kita telah melakukan preprocessing teks yang berupa case folding, dan stopword removal. matriks term-dokumen dibangun dengan menempatkan hasil proses stemming ke dalam baris. setiap baris mewakili kata-kata yang uni, dan setiap kolom mewakili konteks darimana kata-kata tersebut diambil.

berikut code python untuk menghitung Term-document Matrix

```python
vect =TfidfVectorizer(stop_words=list_stopwords,max_features=1000) 
vect_text=vect.fit_transform(dataPTA['abstrak'])
print(vect_text.shape)
print(vect_text)
```

