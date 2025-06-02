# 📘 Cell 1: Kütüphaneler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import lightgbm as lgb
import matplotlib.pyplot as plt
from skgarden import RandomForestQuantileRegressor

# 📘 Cell 2: Veriyi yükle
train = pd.read_csv("house-pi/dataset/dataset.csv")
test = pd.read_csv("house-pi/test.csv")
sample_submission = pd.read_csv("house-pi/sample_submission.csv")

# 📘 Cell 3: Temel sütun isimleri
target_col = "sale_price"
ID_col = "id"

# 📘 Cell 4: Kategorik + sayısal veri işleme
categorical_cols = [
    "city", "zoning", "subdivision", "present_use", "view_rainier", "view_olympics",
    "view_cascades", "view_territorial", "view_skyline", "view_sound", "view_lakewash",
    "view_lakesamm", "view_otherwater", "view_other", "submarket"
]

num_cols = train.select_dtypes(include=np.number).columns.tolist()
num_cols = [col for col in num_cols if col not in [target_col, ID_col]]

for col in categorical_cols:
    if col in train.columns:
        train[col] = train[col].astype(str).fillna("missing")
        test[col] = test[col].astype(str).fillna("missing")

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train[categorical_cols] = encoder.fit_transform(train[categorical_cols])
test[categorical_cols] = encoder.transform(test[categorical_cols])

full_features = list(dict.fromkeys(num_cols + categorical_cols))

train_clean = train.dropna(subset=full_features + [target_col])
X = train_clean[full_features]
y = train_clean[target_col]
test_X = test[full_features]

# 📘 Cell 5: Train-test bölme
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 📘 Cell 6: LightGBM Quantile Regression modeli
def train_lgb_qr(alpha):
    model = lgb.LGBMRegressor(
        objective='quantile',
        alpha=alpha,
        learning_rate=0.03,
        n_estimators=300,
        min_child_samples=20,
        max_depth=7
    )
    model.fit(X_train, y_train)
    return model

model_lgb_lower = train_lgb_qr(0.1)
model_lgb_upper = train_lgb_qr(0.9)

# 📘 Cell 7: QRF modeli
model_qrf = RandomForestQuantileRegressor(random_state=42, n_estimators=100)
model_qrf.fit(X_train, y_train)

qrf_lower = model_qrf.predict(test_X, quantile=10)
qrf_upper = model_qrf.predict(test_X, quantile=90)

# 📘 Cell 8: Ensemble tahminleri
lgb_lower = model_lgb_lower.predict(test_X)
lgb_upper = model_lgb_upper.predict(test_X)

final_pi_lower = (lgb_lower + qrf_lower) / 2
final_pi_upper = (lgb_upper + qrf_upper) / 2

# 📘 Cell 9: Validation Winkler Score
val_lgb_lower = model_lgb_lower.predict(X_val)
val_lgb_upper = model_lgb_upper.predict(X_val)
val_qrf_lower = model_qrf.predict(X_val, quantile=10)
val_qrf_upper = model_qrf.predict(X_val, quantile=90)

final_val_lower = (val_lgb_lower + val_qrf_lower) / 2
final_val_upper = (val_lgb_upper + val_qrf_upper) / 2

def winkler_score(y_true, lower, upper, alpha=0.1):
    score = []
    for yt, l, u in zip(y_true, lower, upper):
        if l <= yt <= u:
            score.append(u - l)
        else:
            penalty = (2 / alpha) * (l - yt) if yt < l else (2 / alpha) * (yt - u)
            score.append((u - l) + penalty)
    return np.mean(score)

val_score = winkler_score(y_val, final_val_lower, final_val_upper)
print("📊 Validation Winkler Score:", val_score)

# 📘 Cell 10: Submission dosyası oluştur
submission = pd.DataFrame({
    "id": test[ID_col],
    "pi_lower": final_pi_lower,
    "pi_upper": final_pi_upper
})

submission.to_csv("submission.csv", index=False)
print("✅ Yeni submission dosyası oluşturuldu.")

# 📘 Cell 11: Feature Importance Görselleştirme
lgb.plot_importance(model_lgb_upper, max_num_features=20)
plt.title("Feature importance (upper bound model)")
plt.show()
