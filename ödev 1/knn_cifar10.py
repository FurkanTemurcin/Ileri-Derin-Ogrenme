import numpy as np
import pickle
import os

# --- 1. VERİ YÜKLEME BÖLÜMÜ ---
def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        X = dict[b'data']
        Y = np.array(dict[b'labels'])
        return X, Y

def load_data(root_path):
    xs = []
    ys = []
    for i in range(1, 6):
        f = os.path.join(root_path, f'data_batch_{i}')
        X, Y = load_cifar10_batch(f)
        xs.append(X)
        ys.append(Y)
    
    X_train = np.concatenate(xs)
    Y_train = np.concatenate(ys)
    X_test, Y_test = load_cifar10_batch(os.path.join(root_path, 'test_batch'))
    
    return X_train, Y_train, X_test, Y_test

# --- 2. K-NN ALGORİTMASI ---
class KNearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X.astype("float32")
        self.y_train = y

    def predict(self, X_test, k=1, metric='l1'):
        X_test = X_test.astype("float32")
        num_test = X_test.shape[0]
        y_pred = np.zeros(num_test, dtype=self.y_train.dtype)

        for i in range(num_test):
            if metric == 'l1':
                # Manhattan Mesafesi
                distances = np.sum(np.abs(self.X_train - X_test[i, :]), axis=1)
            elif metric == 'l2':
                # Euclidean Mesafesi
                distances = np.sqrt(np.sum(np.square(self.X_train - X_test[i, :]), axis=1))

            # En yakın k komşuyu bul ve oyla
            closest_indices = np.argsort(distances)[:k]
            closest_labels = self.y_train[closest_indices]
            counts = np.bincount(closest_labels)
            y_pred[i] = np.argmax(counts)

            if (i + 1) % 50 == 0:
                print(f"İlerleme: {i + 1} / {num_test}")

        return y_pred

# --- 3. ANA ÇALIŞTIRICI (MAIN) ---
if __name__ == "__main__":
    # Klasör yolunu kontrol et
    data_folder = 'cifar-10-batches-py' 
    
    print("Sistem: Veriler yükleniyor...")
    try:
        Xtr, Ytr, Xte, Yte = load_data(data_folder)
        
        # Veriyi düzleştir (Flatten)
        Xtr = Xtr.reshape(Xtr.shape[0], -1)
        Xte = Xte.reshape(Xte.shape[0], -1)

        #Test süresini makul tutmak için ilk 200 örneği alıyoruz
        num_test_samples = 200
        Xte_sub = Xte[:num_test_samples]
        Yte_sub = Yte[:num_test_samples]

        # Kullanıcı etkileşimi
        print("\n--- CIFAR-10 K-NN Sınıflandırıcı ---")
        k_val = int(input("Lütfen k değerini girin (örneğin 3): "))
        metric_choice = input("Mesafe ölçütünü seçin (l1 veya l2): ").strip().lower()

        knn = KNearestNeighbor()
        knn.train(Xtr, Ytr)
        
        print(f"\nİşlem Başladı: k={k_val}, Metrik={metric_choice.upper()}")
        y_pred = knn.predict(Xte_sub, k=k_val, metric=metric_choice)

        accuracy = np.mean(y_pred == Yte_sub)
        print(f"\n>>> TEST SONUCU: %{accuracy * 100:.2f} Doğruluk")

    except FileNotFoundError:
        print(f"HATA: '{data_folder}' klasörü bulunamadı! Lütfen veri setini doğru yere koyduğundan emin ol.")