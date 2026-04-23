# 🧠 İleri Derin Öğrenme Vize Projesi: Alzheimer Sınıflandırma ve XAI

Bu proje, İnönü Üniversitesi Yazılım Mühendisliği Yüksek Lisans "İleri Derin Öğrenme" dersi kapsamında **Muhammed Furkan Temurçin** tarafından geliştirilmiştir. 

Projenin temel amacı, derin öğrenme algoritmaları kullanarak MRI görüntüleri üzerinden Alzheimer hastalığının 4 farklı evresini sınıflandırmak ve Açıklanabilir Yapay Zeka (XAI) teknikleriyle modelin kararlarını medikal olarak doğrulamaktır.

## 🎯 Proje Özeti ve Metrikler
Veri setindeki ciddi sınıf dengesizliğine (Data Imbalance) rağmen, **EfficientNetB3** mimarisi kullanılarak model başarıyla eğitilmiştir.

* **Kullanılan Model:** EfficientNetB3 (Full Fine-Tuning ile tüm katmanlar eğitilmiştir)
* **Genel Başarı (Accuracy):** `%86`
* **Azınlık Sınıf Başarısı:** Sadece 64 görüntüsü olan 'ModerateDemented' sınıfında Data Augmentation ve Class Weights taktikleriyle **%96 F1-Score** ve **1.00 ROC-AUC** değeri elde edilmiştir.

## 🛠️ Kullanılan Yöntemler ve Çözümler
1. **Data Augmentation:** Az veri bulunan sınıfların ezberlenmesini (Overfitting) engellemek için agresif veri çoğaltma uygulanmıştır.
2. **Class Weights (Sınıf Ağırlıkları):** Dengesiz sınıflara matematiksel ceza/ödül sistemi tanımlanmıştır.
3. **Öğrenme Hızı Optimizasyonu:** `ReduceLROnPlateau` ve `1e-5` gibi düşük öğrenme hızlarıyla modelin ince doku detaylarını (atrofi) öğrenmesi sağlanmıştır.

## 🔍 Açıklanabilir Yapay Zeka (Grad-CAM)
Model bir "kara kutu" olarak bırakılmamış, **Grad-CAM** (Gradient-weighted Class Activation Mapping) algoritması uygulanmıştır. 
`grad_cam_final.png` görselinde görülebileceği üzere model; kararlarını verirken beynin rastgele noktalarına değil, medikal literatürle uyuşan kortikal atrofi ve ventrikül boşluklarına odaklanmaktadır.

## 📂 Klasör İçeriği
* `Furkan_Temurcin_Alzheimer_Model.ipynb`: Tüm eğitim, test ve Grad-CAM aşamalarını içeren kaynak kod.
* `egitim_gecmisi_final.csv`: Epoch bazlı Accuracy ve Loss değerleri.
* `final_matrix.png`: Modelin karmaşıklık matrisi (Confusion Matrix).
* `final_roc.png`: Sınıfların çoklu ROC-AUC eğrileri.
* `grad_cam_final.png`: XAI (Açıklanabilir Yapay Zeka) ısı haritası örneği.
