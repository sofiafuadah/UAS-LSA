# LSA (Latent Semantic Analysis)

Algoritma LSA (latent Semantic Analysis) adalah salah satu algoritma yang dapat digunakan untuk menganalisa hubungan antara sbeuah frase/kalimat dengan sekumpulan dokumen.

Contoh yang dibahas kali ini adalah mengenai penentuan urutan peringkat data berdasarkan query yang digunakan.

# Algoritma LSA

langkah-langkah algoritma LSA dalam preprocessing teks

# Menghitung Term-document Matrix

sebelum meghitung term-document matrix (TF-IDF) kita telah melakukan preprocessing teks yang berupa case folding, dan stopword removal. matriks term-dokumen dibangun dengan menempatkan hasil proses stemming ke dalam baris. setiap baris mewakili kata-kata yang uni, dan setiap kolom mewakili konteks darimana kata-kata tersebut diambil.

cara menghitung term-document matrix dapat dirumuskan dengan 
$$
\begin{aligned}.\\ W_{ij}=tfi,j\times \log \dfrac{N}{d_{fj}}\\ .\end{aligned}
$$

Wij = score Term-document matrix (TF-IDF)

Tfi,j = term dokumen

N = total dokumen

Dfj = dokumen



pada pyhton dapat dijalankan program sebagai berikut:

```{tableofcontents}
vect =TfidfVectorizer(stop_words=list_stopwords,max_features=1000) 
vect_text=vect.fit_transform(dataPTA['Abstrak'])
print(vect_text.shape)
print(vect_text)
```

# Singular Value Decomposition

Singular Value Decomposition (SVD) adalah salah satu teknik reduksi dimensi yang bermanfaat untuk memperkecil nilai kompleksitas dalam pemroses term-document matrix. SVD merupakan teorema aljabar linier yang menyebutkan bahwa persegi panjang dari term-document matrix dapat dipecah/didekomposisikan menjadi tiga matriks, yaitu :

1. Matriks orotgunal U

2. Matriks diagonal S

3. Transpose dari matriks ortogonal V 

   yang dirumuskan dengan :
   $$
   A_{mn}=U_{mm}\times S_{mn}\times V_{nn}^{T}
   $$
   Amn = matriks awal

   Umm = Matriks ortogonal U

   Smm = Matriks diagonal S

   VTnn = transpose matriks ortogonal V


Hasil dari proses SVD adalah vektor yang akan digunakan untuk menghitung similaritasnya dengan pendekatan cosine similarity



implementasi SVD pada pyhton yaitu :

```python
from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)
print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)
```

```python
l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)
```

```python
print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)
```

# Mengekstrak Topik dan Term

setelah dilakukan beberapa tahapan seperti diatas, selanjutnya kita akan mengekstrak topik dokumen. pada percobaan kali ini kita akan mencoba mengekstrak sebanyak 10 topik.

```python
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")
```

