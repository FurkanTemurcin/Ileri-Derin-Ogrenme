# Modern Evrişimli Sinir Ağları: AlexNet'ten Toplu Normalleştirmeye

Bu klasör, İleri Derin Öğrenme dersi kapsamında Bölüm 7 (Modern Evrişimli Sinir Ağları) uygulamalarını içermektedir. Bütün mimariler, d2l kütüphanesine bağımlı kalınmadan saf **TensorFlow/Keras** ile inşa edilmiş ve Fashion-MNIST veri seti üzerinde eğitilmiştir.

##  İçerik ve Gerçekleştirilen Deneyler

Aşağıdaki mimariler sıfırdan kodlanmış, her birinin matris çıktı boyutları test edilmiş ve grafiksel eğitim sonuçları alınmıştır. *(Not: Eğitimlere ait parametre analizi ve katman çıktı özetleri doğrudan `.ipynb` dosyasının içindeki hücre çıktılarında mevcuttur).*

1. **AlexNet (7.1):** 11x11 filtreler, ReLU aktivasyonu ve Dropout matematiksel ölçeklemesi ile derinliğin ispatı.
2. **VGG (7.2):** Blok mimarisi konsepti ve 3x3 evrişimlerin getirdiği parametre verimliliğinin incelenmesi. (Sistem belleği için `ratio=4` ile küçültülmüş versiyon kullanılmıştır).
3. **NiN - Ağ İçinde Ağ (7.3):** Tam Bağlı (Dense) katmanların silinip yerine 1x1 evrişim ve Global Average Pooling (GAP) eklenmesiyle parametre sayısının 46 Milyondan **1.9 Milyona** düşürüldüğünün ispatı.
4. **GoogLeNet / Inception (7.4):** Farklı boyutlardaki evrişimlerin paralel çalıştırılması ve Darboğaz (Bottleneck) katmanlarının verimlilik analizi.
5. **Toplu Normalleştirme / Batch Norm (7.5):** Ağın kararlılığını test etmek amacıyla `learning_rate = 1.0` gibi absürt bir değerle yapılan stres testi ve modelin  (Gamma) / (Beta) değerlerini başarıyla öğrenerek çökmekten kurtulması.

## 📁 Klasör Yapısı
* `Modern_CNN_Uygulamalari.ipynb` : Bütün mimarilerin çalıştırılabilir kaynak kodları ve eğitim (konsol) çıktıları.
* `/Grafikler` : Her bir modelin eğitim (Accuracy/Loss) eğrilerini içeren grafik dosyaları.

**Geliştirici:** Muhammed Furkan Temurçin
