# Pengertian K-Means

Algoritma _K-Means_ digunakan untuk memecahkan masalah dalam pengelompokan pada _teks_. Algoritma _K-Means_ dikelompokkan untuk menghasilkan suatu informasi yang terdapat dalam setiap kelompok teks. Menurut Widyawati (2010 h, 7) mengungkapkan bahwa : “Algoritma _K-Means_ membutuhkan parameter input sebanyak _k_ dan membagi sekumpulan n objek (data) menjadi _k_ kelompok (_cluster_), sehingga anggota dalam satu kelompok (_cluster_) memiliki tingkat kemiripan yang tinggi, dan juga memiliki tingkat kemiripan yang sangat rendah dengan kelompok (_cluster_) lain. Tingkat kemiripan suatu anggota terhadap suatu kelompok (_cluster_) diukur berdasarkan kedekatan objek (data) dengan nilai rata-rata dari kelompok (_cluster_) lain atau yang biasa disebut dengan sentroid pusat.

# Proses Clustering K-Means pada Teks Mining

1. Import Library

   untuk melakukan proses clustering k-means kita akan menggunakan library sklearn.cluster dan importkan KMeans

   ```python
   from sklearn.cluster import KMeans
   ```

2. lakukan text preprocessing

   untuk melakukan text prepocessing, lakukan langkah seperti yang sudah dijelaskan sebelumnya yaitu mulai dari melakukan case folding, tokenizing, stopword, stemming, dan melakukan perhitungan TF-IDF

3. lakukan proses K-Means

   ​

# Algoritma K-Means

Algoritma pengelompokan teks menggunakan metode* K-Means * adalah sebagai berikut :

1.  Menentukan banyaknya kelompok (_cluster_), dimana kelompok telah ditentukan sebanyak 4 kelompok. nilai random digunakan untuk menentukan nilai centroid awal

    ```python
    true_k = 4
    model = KMeans(n_clusters= true_k, init='k-means++', max_iter=100, n_init=1, random_state=0)
    model.fit(vect_text)
    ```

2.  kemudian objek vektor yang telah didapatkan dari proses pembobotan dialokasikan, dan selanjutnya menentukan centroidnya secara random.

    ```python
    clusters = model.predict(vect_text)
    ```

3.  setelah centroid ditentukan maka selanjutnya menghitung jarak antara 2 vektor, dalam hal ini adalah jarak antara term dan centroid dengan menggunakan rumus Euclidean Distance, seperti pada rumus berikut ini :

    ```python
    print("kata teratas per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vect.get_feature_names()
    for i in range(4):
        print("cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind]),
        print
    ```

4.  jika proses berubah lagi proses kembali ke langkah 3 dengan penentuan posisi centroid.

    berikut ini adalah cara menampilkan hasil sentroid akhir

    ```python
    print(model.cluster_centers_)
    ```
