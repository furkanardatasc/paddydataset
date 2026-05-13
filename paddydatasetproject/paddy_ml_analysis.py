"""
==============================================================
PADDY VERİ SETİ - MAKİNE ÖĞRENMESİ ALGORİTMALARI KARŞILAŞTIRMASI
==============================================================
Kaynak : UCI Machine Learning Repository - Paddy Dataset (ID: 1186)
DOI    : https://doi.org/10.24432/C55W3J
Hedef  : Paddy çeşidini (Variety) tahmin etmek
         3 Sınıf: CO_43 | ponmani | delux ponni
Örnek  : 2789 | Özellik: 45 | Bölme: %80 Eğitim / %20 Test

Kullanılan Algoritmalar:
  1. Rastgele Orman    (Random Forest)
  2. Karar Ağacı       (Decision Tree)
  3. K-En Yakın Komşu (KNN, k=5)
  4. Lineer Regresyon  (Logistic Regression)
  5. Basit Bayes       (Naive Bayes)
  6. CNN               (MLP 128-64-32, relu)
  7. YSA               (MLP 64-32, tanh)
  8. SVM               (RBF Kernel)

KULLANIM:
  pip install pandas scikit-learn numpy
  python paddy_ml_analysis.py

  'paddydataset.csv' dosyası aynı klasörde olmalıdır.
  İndir: https://archive.ics.uci.edu/dataset/1186/paddy+dataset
==============================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. VERİYİ YÜKLE
# ─────────────────────────────────────────────
print("=" * 65)
print("  PADDY VERİ SETİ — MAKİNE ÖĞRENMESİ ANALİZİ")
print("=" * 65)

try:
    df = pd.read_csv('paddydataset.csv')
except FileNotFoundError:
    raise FileNotFoundError(
        "'paddydataset.csv' bulunamadı.\n"
        "İndir: https://archive.ics.uci.edu/dataset/1186/paddy+dataset"
    )

print(f"\n  Toplam örnek : {df.shape[0]}")
print(f"  Özellik sayısı: {df.shape[1]}")
print(f"  Hedef (Variety): {df['Variety'].value_counts().to_dict()}")

# ─────────────────────────────────────────────
# 2. ÖN İŞLEME
# ─────────────────────────────────────────────
df_enc = df.copy()
for col in df_enc.select_dtypes(include='object').columns:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

X = df_enc.drop(columns=['Variety'])
y = df_enc['Variety']

# %80 Eğitim / %20 Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Ölçekleme (mesafeye / gradyana duyarlı modeller için)
sc = StandardScaler()
X_tr_sc = sc.fit_transform(X_train)
X_te_sc  = sc.transform(X_test)

print(f"\n  Eğitim seti : {len(X_train)} örnek")
print(f"  Test seti   : {len(X_test)} örnek  (80/20 bölme)")

# ─────────────────────────────────────────────
# 3. ALGORİTMALAR
# ─────────────────────────────────────────────
results = {}

def evaluate(name, model, Xtr, Xte, verbose=True):
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    if verbose:
        print(f"\n  {name}")
        print(classification_report(y_test, y_pred,
              target_names=['CO_43','delux ponni','ponmani'],
              zero_division=0))
    return acc

print("\n" + "─" * 65)
print("  ALGORİTMA DETAYLARI")
print("─" * 65)

evaluate("Rastgele Orman",
         RandomForestClassifier(n_estimators=100, random_state=42),
         X_train, X_test)

evaluate("Karar Agaci",
         DecisionTreeClassifier(random_state=42),
         X_train, X_test)

evaluate("K-En Yakin Komsu (k=5)",
         KNeighborsClassifier(n_neighbors=5),
         X_tr_sc, X_te_sc)

evaluate("Lineer Regresyon",
         LogisticRegression(max_iter=1000, random_state=42),
         X_tr_sc, X_te_sc)

evaluate("Basit Bayes",
         GaussianNB(),
         X_tr_sc, X_te_sc)

evaluate("CNN (128-64-32 relu)",
         MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                       activation='relu', solver='adam',
                       max_iter=500, random_state=42),
         X_tr_sc, X_te_sc)

evaluate("YSA (64-32 tanh)",
         MLPClassifier(hidden_layer_sizes=(64, 32),
                       activation='tanh', solver='adam',
                       max_iter=500, random_state=42),
         X_tr_sc, X_te_sc)

evaluate("SVM (RBF Kernel)",
         SVC(kernel='rbf', random_state=42),
         X_tr_sc, X_te_sc)

# ─────────────────────────────────────────────
# 4. ÖZET TABLO
# ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ÖZET TABLO — Paddy Veri Seti Doğruluk Tahmin Oranları")
print("=" * 65)
print(f"  {'Değerlendirme Kriterleri':<32} {'Doğruluk Tahmin Oranı':>22}")
print("  " + "─" * 55)

best = max(results, key=results.get)
for algo, acc in results.items():
    flag = "  ◄ EN İYİ" if algo == best else ""
    print(f"  {algo:<32} {acc:>22.4f}{flag}")

print("  " + "─" * 55)
print(f"\n  En İyi Algoritma : {best}")
print(f"  Doğruluk         : {results[best]:.4f}  ({results[best]*100:.2f}%)")

# ─────────────────────────────────────────────
# 5. KAYDET
# ─────────────────────────────────────────────
pd.DataFrame(
    list(results.items()),
    columns=['Degerlendirme Kriterleri', 'Dogruluk Tahmin Orani']
).to_csv('paddy_sonuclar.csv', index=False, encoding='utf-8-sig')

print("\n  Sonuçlar → 'paddy_sonuclar.csv' kaydedildi.")
print("=" * 65)
