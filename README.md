# Enerji Tüketimi Tahmini - LSTM

Bu proje, geçmiş enerji tüketim verilerini kullanarak **gelecek 24 saatlik enerji tüketimini LSTM (Long Short-Term Memory)** modeli ile tahmin etmeyi amaçlamaktadır.

Proje kapsamında `household_power_consumption.txt` veri seti temizlenmiş, saatlik enerji tüketimine dönüştürülmüş, LSTM modelinin kullanabileceği zaman serisi verileri hazırlanmış ve model eğitilerek gelecek 24 saat için tahmin yapılmıştır.

## Proje Akışı

```text
household_power_consumption.txt
            ↓
     Veri temizleme
            ↓
       df_hourly.csv
            ↓
    MinMaxScaler ile ölçekleme
            ↓
     Sliding Window
            ↓
   X_train / X_test
   y_train / y_test
            ↓
       LSTM Modeli
            ↓
      lstm_model.keras
            ↓
    Gelecek 24 Saat Tahmini
            ↓
     Gerçek vs Tahmin Grafiği
```

## Kullanılan Teknolojiler

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- TensorFlow / Keras
- LSTM

## Veri Seti

Projede `household_power_consumption.txt` veri seti kullanılmaktadır.

Veri setindeki temel değişkenlerden biri:

- `Global_active_power`: Global aktif güç tüketimi (kW)

Veri önce tarih ve saat bilgileri birleştirilerek zaman serisine dönüştürülür. Ardından saatlik ortalama enerji tüketimi hesaplanır.

Sonuç:

```text
df_hourly.csv
```

dosyasına kaydedilir.

## Veri Hazırlama

Saatlik veri `MinMaxScaler` kullanılarak 0-1 aralığında ölçeklendirilir.

LSTM modeline geçmiş **24 saatlik veri** giriş olarak verilir ve bir sonraki saat tahmin edilir.

Örneğin:

```text
Saat 1  ─┐
Saat 2   │
Saat 3   │
...      ├──→ LSTM ──→ Saat 25 tahmini
Saat 24 ─┘
```

Bu işlem sliding window yöntemiyle gerçekleştirilir.

Veriler %80 eğitim ve %20 test olarak ayrılır.

Oluşturulan dosyalar:

```text
X_train.npy
y_train.npy
X_test.npy
y_test.npy
scaler.save
```

## LSTM Modeli

Model aşağıdaki temel yapıya sahiptir:

```text
Input
  ↓
LSTM (64 units)
  ↓
Dense (1)
```

LSTM katmanında:

- 64 nöron
- `tanh` aktivasyon fonksiyonu

kullanılmıştır.

Model:

```text
Optimizer: Adam
Loss: Mean Squared Error (MSE)
Epochs: 10
Batch Size: 32
```

Eğitim sırasında `EarlyStopping` kullanılarak doğrulama kaybı belirli sayıda epoch boyunca iyileşmediğinde eğitim durdurulur.

Model:

```text
lstm_model.keras
```

dosyasına kaydedilir.

## Gelecek 24 Saat Tahmini

Eğitilen model kullanılarak gelecek 24 saat için tahmin yapılır.

Tahmin sürecinde:

1. Son 24 saatlik gerçek veri alınır.
2. Daha önce eğitilmiş `MinMaxScaler` ile ölçeklendirilir.
3. LSTM modeline gönderilir.
4. Bir sonraki saat tahmin edilir.
5. Tahmin tekrar giriş dizisine eklenir.
6. İlk değer diziden çıkarılır.
7. İşlem 24 kez tekrarlanır.
8. Tahminler tekrar gerçek ölçeğe dönüştürülür.

Sonuç olarak gerçek ve tahmin edilen değerler aynı grafik üzerinde gösterilir.

## Proje Dosya Yapısı

## Kurulum

Öncelikle bir Python virtual environment oluşturulması önerilir:

Environment aktifleştirildikten sonra gerekli paketler kurulabilir:

```bash
python -m pip install numpy pandas matplotlib scikit-learn joblib tensorflow optree
```

Keras, TensorFlow ile birlikte kullanılmaktadır.

## Çalıştırma

### 1. Veriyi temizle

```bash
python diyipro_load_and_clean.py
```

Bu işlem sonucunda:

```text
df_hourly.csv
```

oluşturulur.

### 2. Eğitim ve test verilerini oluştur

```bash
python diyipro_prepare_data.py
```

Bu işlem sonucunda:

```text
X_train.npy
y_train.npy
X_test.npy
y_test.npy
scaler.save
```

dosyaları oluşturulur.

### 3. LSTM modelini eğit

```bash
python diyipro_lstm_train.py
```

Eğitim tamamlandığında:

```text
lstm_model.keras
```

oluşturulur ve eğitim/doğrulama kayıp grafiği gösterilir.

### 4. Gelecek 24 saati tahmin et

```bash
python diyipro_lstm_predict.py
```

Bu işlem gerçek ve tahmin edilen gelecek 24 saatlik enerji tüketimini karşılaştıran bir grafik oluşturur.

## Model Kayıp Grafiği

Model eğitimi sırasında eğitim ve doğrulama kayıpları takip edilir.

```text
Loss
 │
 │\
 │ \
 │  \
 │   \____
 │
 └──────────────── Epoch
```

Eğitim kaybının ve doğrulama kaybının zaman içerisindeki değişimi modelin öğrenme sürecini incelemek için kullanılmaktadır.

## Tahmin Grafiği

Son aşamada:

- Gerçek gelecek 24 saatlik tüketim
- LSTM tarafından tahmin edilen gelecek 24 saatlik tüketim

aynı grafik üzerinde karşılaştırılır.

Bu sayede modelin gerçek değerleri ne kadar yakından takip ettiği görsel olarak incelenebilir.

## Notlar

Modelin performansı kullanılan veri setine, zaman aralığına, pencere uzunluğuna ve model hiperparametrelerine bağlıdır.

Bu projede pencere uzunluğu:

```text
24 saat
```

olarak belirlenmiştir.

Daha iyi sonuçlar için ileride farklı pencere uzunlukları, daha fazla LSTM katmanı, farklı nöron sayıları ve farklı hiperparametreler denenebilir.

## Sonuç

Bu proje ile zaman serisi verileri kullanılarak LSTM tabanlı bir enerji tüketimi tahmin sistemi oluşturulmuştur.

Proje temel olarak:

Veri Toplama
     ↓
Veri Temizleme
     ↓
Saatlik Veriye Dönüştürme
     ↓
Veri Ölçekleme
     ↓
Sliding Window
     ↓
LSTM Eğitimi
     ↓
24 Saatlik Tahmin
     ↓
Gerçek / Tahmin Karşılaştırması
```

adımlarından oluşmaktadır.
