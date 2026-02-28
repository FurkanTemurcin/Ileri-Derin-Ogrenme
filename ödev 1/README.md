\# Ödev 1: CIFAR-10 Veri Seti Üzerinde K-NN Sınıflandırma ve Hiperparametre Optimizasyonu



Bu proje, CIFAR-10 veri seti üzerinde K-Nearest Neighbor (K-NN) algoritmasını herhangi bir hazır kütüphane (scikit-learn vb.) kullanmadan, "from scratch" (açık kod) mantığıyla gerçekleştiren bir akademik çalışmadır.



\## 🚀 Proje Özellikleri

\- Veri seti diskten `pickle` kütüphanesi ile okunmaktadır.

\- \*\*L1 (Manhattan)\*\* ve \*\*L2 (Euclidean)\*\* mesafe ölçütleri vektörize edilerek implement edilmiştir.

\- 3072 boyutlu (32x32x3) ham piksel verileri üzerinde doğrudan sınıflandırma yapılmıştır.

\- Kullanıcıdan dinamik olarak \*\*k\*\* ve \*\*metrik\*\* parametrelerini alacak şekilde tasarlanmıştır.



\## 📊 Deney Sonuçları ve Hiperparametre Analizi

Model performansı, 200 adet test örneği üzerinde farklı $k$ değerleri ve mesafe metrikleri kullanılarak 16 farklı senaryoda test edilmiştir.



| k Değeri | Mesafe Metriği | Doğruluk (Accuracy) | Durum |

| :--- | :--- | :--- | :--- |

| 1 | L1 | %36.50 | Başlangıç |

| 1 | L2 | %35.50 | - |

| 3 | L1 | %33.50 | Yerel Minimum |

| 3 | L2 | %34.50 | - |

| 5 | L1 | %37.00 | Yükseliş |

| 5 | L2 | %34.50 | - |

| 7 | L1 | %37.00 | Plato |

| 7 | L2 | %32.00 | Kritik Düşüş |

| 9 | L1 | %38.00 | İyileşme |

| 9 | L2 | %37.00 | - |

| \*\*11\*\* | \*\*L1\*\* | \*\*%38.50\*\* | \*\*Zirve (Optimal)\*\* |

| \*\*11\*\* | \*\*L2\*\* | \*\*%38.50\*\* | \*\*Zirve (Optimal)\*\* |

| 13 | L1 | %35.50 | Düşüş (Aşırı Genelleme) |

| 13 | L2 | %37.00 | - |

| 15 | L1 | %35.50 | Düşüş Onaylandı |

| 15 | L2 | %35.50 | Düşüş Onaylandı |



\## 🧐 Teknik Değerlendirme

\- \*\*Optimal K Değeri:\*\* Yapılan testler sonucunda modelin en yüksek genelleme kapasitesine \*\*k=11\*\* noktasında ulaştığı görülmüştür.

\- \*\*Mesafe Metrikleri:\*\* Genel trendde \*\*L1 (Manhattan)\*\* mesafesinin, ham piksellerdeki gürültüye karşı \*\*L2 (Euclidean)\*\* mesafesine göre daha dayanıklı (robust) olduğu saptanmıştır.

\- \*\*Gürültü ve Underfitting:\*\* k=1 ve k=3 gibi düşük değerlerdeki istikrarsızlık, verideki gürültüden kaynaklanırken; k=13'ten sonraki düşüşler modelin ayırt edici detayları kaybettiğini (underfitting) işaret etmektedir.



$$d\_1(I\_1, I\_2) = \\sum\_{p} |I\_1^p - I\_2^p| \\quad \\text{vs} \\quad d\_2(I\_1, I\_2) = \\sqrt{\\sum\_{p} (I\_1^p - I\_2^p)^2}$$

