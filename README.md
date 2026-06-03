# Data Science Project 36 — Kalp Hastalığı Riski (Random Forest + Feature Importance)

**Modül**: ML-04 (Sınıflandırma 2) • **Süre**: 3-4 saat

## 🎯 Proje Senaryosu

Bir hastanede **klinik karar destek sistemi** kuruyorsun. Doktorlar acil serviste hastalardan hızlıca ölçüm alıyor: yaş, tansiyon, kolesterol, göğüs ağrısı tipi, EKG sonucu, maksimum kalp hızı vb. Senden bu ölçümlere bakarak hastanın **kalp hastalığı riski** taşıyıp taşımadığını tahmin eden bir model istiyorlar.

Ama doktor sadece "var / yok" cevabı istemiyor. Asıl merak ettiği şu:

> **"Hangi faktör kalp hastalığını en çok belirliyor?"**

İşte burada **Random Forest** parlıyor. Hem güçlü bir model (yüzlerce ağacın oyu), hem de **`feature_importances_`** sayesinde "hangi ölçüm en kritik?" sorusuna sayısal cevap veriyor. Bu, doktora doğrudan **business insight** (klinik içgörü) sağlıyor — örneğin "göğüs ağrısı tipi ve max kalp hızı en belirleyici" gibi.

Bu projede ML-04 dersinde öğrendiklerini uygulayacaksın:
- ✅ **RandomForestClassifier** — ensemble (topluluk) gücü
- ✅ **`feature_importances_`** + barplot → hangi faktör kritik
- ✅ **`n_estimators` / `max_depth`** hiperparametre tuning
- ✅ **Decision Tree vs Random Forest** karşılaştırma (RF neden daha kararlı)
- ✅ **OOB (Out-Of-Bag) skoru** → ücretsiz validation
- ✅ Ağaç tabanlı modellerde **scaling neden GEREKMEZ**

## 📦 Proje Kurulumu

```bash
# Fork + clone
git clone <your-fork-url>
cd data-science-project-36

# Virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate          # Windows

# Dependencies
pip install -r requirements.txt

# Auto test runner (dosya değişince çalışır)
python watch.py

# Manuel test
pytest tests/test_question.py -v
```

## 🔑 Kaizu Bağlantısı — `kaizu_config.py`

Skorunun Kaizu hesabına yazılması için **`kaizu_config.py`** dosyasını aç ve **`USER_ID`** alanını kendi user_id'nle değiştir:

```python
USER_ID = 0      # ← Kaizu profilinden alıp buraya yaz
PROJECT_ID = 716 # ← Bu projeye ait, dokunma
```

User_id'ni Kaizu profilinden bulabilirsin (Profile → Settings → User ID).

Skor göndermek için tüm testleri toplu çalıştırmalısın:

```bash
python tests/test_question.py
```

Bu komut tüm testleri çalıştırır, **passed/total oranını otomatik Kaizu'ya gönderir**. Geliştirme sırasında `pytest -v` kullanmaya devam edebilirsin (skor göndermez).

## 📊 Veri Seti

**Kaynak**: UCI Machine Learning Repository — *Heart Disease Data Set*, **Cleveland** kliniği (Hungarian Institute of Cardiology / Cleveland Clinic Foundation, 1988). En çok kullanılan medikal ML benchmark'larından biridir.

**Dosya**: `data/heart.csv` (repo içinde, indirmen gerekmez)

**Boyut**: 303 hasta × 13 feature + 1 hedef sütun
**Hedef dağılımı**: 164 sağlıklı (`target=0`) / 139 kalp hastalığı (`target=1`) → görece **dengeli** sınıflar (~%46 hasta)

> Not: Orijinal UCI verisinde hedef sütun `num` 0–4 arası bir **şiddet** skorudur (0 = hastalık yok, 1–4 = farklı derecelerde damar tıkanıklığı). Bu projede klinik "var/yok" kararı için **binary** hale getirildi: `target = 1 if num > 0 else 0`.

### Feature Dictionary

| # | Sütun | Ne ölçer? | Tip | Değerler / Aralık |
|---|-------|-----------|-----|--------|
| 1 | `age` | Yaş (yıl) | sürekli | 29–77 |
| 2 | `sex` | Cinsiyet | kategorik-kodlu | 1 = erkek, 0 = kadın |
| 3 | `cp` | **Göğüs ağrısı tipi** (chest pain) | kategorik-kodlu | 1 = tipik angina, 2 = atipik, 3 = anginal olmayan ağrı, 4 = asemptomatik |
| 4 | `trestbps` | Dinlenme **tansiyonu** (mm Hg) | sürekli | 94–200 |
| 5 | `chol` | Serum **kolesterol** (mg/dl) | sürekli | 126–564 |
| 6 | `fbs` | Açlık kan şekeri > 120 mg/dl | kategorik-kodlu | 1 = evet, 0 = hayır |
| 7 | `restecg` | Dinlenme EKG sonucu | kategorik-kodlu | 0 = normal, 1 = ST-T anomalisi, 2 = sol ventrikül hipertrofisi |
| 8 | `thalach` | Egzersizde ulaşılan **maksimum kalp hızı** | sürekli | 71–202 |
| 9 | `exang` | Egzersize bağlı angina (göğüs ağrısı) | kategorik-kodlu | 1 = var, 0 = yok |
| 10 | `oldpeak` | Egzersize bağlı ST depresyonu (EKG) | sürekli | 0.0–6.2 |
| 11 | `slope` | Egzersiz ST segment eğimi | kategorik-kodlu | 1 = yukarı, 2 = düz, 3 = aşağı |
| 12 | `ca` | Floroskopiyle boyanan ana **damar sayısı** | kategorik-kodlu | 0–3 |
| 13 | `thal` | Talasemi / kan akışı testi | kategorik-kodlu | 3 = normal, 6 = sabit kusur, 7 = geri dönüşlü kusur |
| — | `target` | **Hedef** — 1: kalp hastalığı var, 0: yok | int | 0 / 1 |

### Örnek Satırlar

```
age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target
63,1,1,145,233,1,2,150,0,2.3,3,0,6,0
67,1,4,160,286,0,2,108,1,1.5,2,3,3,1
67,1,4,120,229,0,2,129,1,2.6,2,2,7,1
37,1,3,130,250,0,0,187,0,3.5,3,0,3,0
41,0,2,130,204,0,2,172,0,1.4,1,0,3,0
```

### Domain Notu

Bu **mixed feature** (karışık tipli) bir veri setidir: bazı sütunlar sürekli sayısal (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`), bazıları **kategorik ama sayı ile kodlanmış** (`cp`, `restecg`, `slope`, `thal`, `ca`, `sex`, `fbs`, `exang`). Random Forest gibi **ağaç tabanlı** modeller bu karışımı doğrudan işleyebilir — değerlerin büyüklük sırası değil, eşik bölmeleri (split) önemlidir.

Bu yüzden bu projede **scaling (StandardScaler) GEREKMEZ**: Random Forest mesafe tabanlı değildir (KNN'in aksine), her ağaç feature'ları ayrı ayrı eşiklerle böler. Bir feature'ı ölçeklemek ağacın kararını değiştirmez.

Orijinal veride `ca` ve `thal` sütunlarında 6 hastada eksik değer (`?`) vardı; satır sayısını 303'te tutmak için bunlar o sütunun en sık değeriyle (mode) dolduruldu.

## 📋 Görevler (`tasks/task_manager.py`)

`task_manager.py` dosyasındaki **14 fonksiyonu** sırayla doldur. Her fonksiyonun altındaki testler pass olana kadar düzenle.

1. **`load_heart_data(path)`** — `data/heart.csv` oku, X (feature) ve y (target) döndür
2. **`explore_data(X, y)`** — örnek/feature sayısı, pozitif/negatif dağılım, feature isimleri
3. **`split_data(X, y)`** — train/test böl (test_size=0.2, stratify, random_state=42)
4. **`train_random_forest(X_train, y_train, n_estimators)`** — RF kur + eğit
5. **`evaluate_model(model, X_test, y_test)`** — accuracy, precision, recall, f1, confusion matrix
6. **`get_feature_importances(model, feature_names, top_n)`** — en önemli feature'lar (sıralı)
7. **`plot_feature_importances(model, feature_names)`** — yatay bar grafik (barh)
8. **`tune_n_estimators(...)`** — farklı ağaç sayıları için accuracy
9. **`compare_dt_vs_rf(...)`** — tek Decision Tree vs Random Forest
10. **`get_oob_score(X_train, y_train, n_estimators)`** — `oob_score=True` → out-of-bag accuracy
11. **`predict_patient(model, patient_row)`** — tek hasta tahmini + risk seviyesi
12. **`cross_validate_rf(X, y, cv)`** — cross-validation mean/std
13. **`simplify_with_top_features(...)`** — sade model (top 5 feature) aynı performansta mı?
14. **`run_pipeline()`** — tüm akışı uçtan uca çalıştır

## 🎓 Öğrenme Hedefleri

Bu projeyi bitirdiğinde:
- [x] **RandomForestClassifier** kurup eğitebileceksin (ensemble mantığı)
- [x] **`feature_importances_`** ile "hangi faktör kritik?" sorusunu yanıtlayacaksın
- [x] Feature importance'ı **barplot** ile görselleştireceksin
- [x] **`n_estimators`** (ağaç sayısı) ve **`max_depth`** tuning'in etkisini göreceksin
- [x] Tek **Decision Tree vs Random Forest** farkını ölçeceksin (RF neden daha kararlı/yüksek)
- [x] **OOB skoru** ile ayrı validation seti ayırmadan genelleme gücünü tahmin edeceksin
- [x] Ağaç tabanlı modellerde **scaling neden gereksiz** olduğunu anlayacaksın
- [x] **Sade model** (az feature) vs tam model trade-off'unu değerlendireceksin (klinikte daha az ölçüm = daha ucuz)

## 🧪 Testler

Test dosyası: `tests/test_question.py` (20+ test)

Öne çıkanlar:
- Veri yükleme + shape (303 × 13) + binary target
- Class dağılımı (164 / 139)
- Stratified split (test oranı genel orana yakın)
- RF tipi + `n_estimators=100`
- `test_run_pipeline_full` → **test accuracy > 0.75**
- Feature importance **sıralı** + klinik olarak anlamlı (cp, thalach, ca, thal, oldpeak üstte)
- RF >= DT (ensemble en az tek ağaç kadar iyi)
- OOB skoru makul aralıkta (0.6–1.0)

## 📊 Beklenen Sonuçlar

```
Örnek: 303 hasta → Train 242, Test 61 (stratify)
Test accuracy (RF, n=100): ~0.85-0.90
En kritik faktör (feature importance #1): thalach (max kalp hızı) / cp / thal
OOB score: ~0.78-0.82
Decision Tree accuracy: ~0.74   →   Random Forest accuracy: ~0.88
  → RF, tek ağaçtan belirgin şekilde daha kararlı (+~0.10-0.15)
Sade model (top 5 feature): tam modele çok yakın accuracy
```

## 💡 İpuçları

- Random Forest **scaling istemez** → veriyi olduğu gibi ver
- `feature_importances_` → 0–1 arası, toplamı 1; büyük = daha belirleyici
- `oob_score=True` ver, `.fit()` sonrası `.oob_score_` oku
- DT vs RF: aynı `random_state=42` ile karşılaştır, adil olsun
- `predict_proba(X)[0, 1]` → pozitif sınıf (hastalık var) olasılığı
- DataFrame'den sütun alt kümesi seçmek için: `X_train[['cp', 'thalach', ...]]`
- `random_state=42` her yerde — tekrarlanabilirlik

## 🚫 Dikkat

- `tests/test_question.py` dosyasını **değiştirme**
- `random_state=42` değerini değiştirme (testler fail olur)
- `_solution/` klasörü repoda yok (DB'de saklanır, dersin haftası geçince açılır)
- Dokunabileceğin **2 dosya**: `tasks/task_manager.py` (kodu yaz) + `kaizu_config.py` (sadece USER_ID)
