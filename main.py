# 📘 Cell 1: Kütüphaneler
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
import matplotlib.pyplot as plt

# 📘 Cell 2: Veriyi yükle
train = pd.read_csv("house-pi/dataset/dataset.csv")
test = pd.read_csv("house-pi/test.csv")
sample_submission = pd.read_csv("house-pi/sample_submission.csv")

# 📘 Cell 3: Temel sütun isimleri
target_col = "sale_price"
ID_col = "id"

# 📘 Cell 4: Feature engineering + encoding
train = train[train[target_col] < 1_500_000]  # outlier filtreleme

train["sale_date"] = pd.to_datetime(train["sale_date"])
test["sale_date"] = pd.to_datetime(test["sale_date"])
train["sale_year"] = train["sale_date"].dt.year
train["sale_month"] = train["sale_date"].dt.month
test["sale_year"] = test["sale_date"].dt.year
test["sale_month"] = test["sale_date"].dt.month

train["building_age"] = train["sale_year"] - train["year_built"]
test["building_age"] = test["sale_year"] - test["year_built"]

train["price_per_sqft"] = train[target_col] / (train["sqft"] + 1)
test["price_per_sqft"] = 0  # bilinmiyor

train["living_area_per_room"] = train["sqft"] / (train["beds"] + 1)
test["living_area_per_room"] = test["sqft"] / (test["beds"] + 1)

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
X = train_clean[full_features].values
y = np.log1p(train_clean[target_col].values)  # log dönüşümü
test_X = test[full_features].values

# Güvenli expm1 fonksiyonu: uç değerleri kırpar
def safe_expm1(preds, clip_min=-10, clip_max=15):
    clipped = np.clip(preds, clip_min, clip_max)
    return np.expm1(clipped)

# Winkler score fonksiyonu
def winkler_score(y_true, lower, upper, alpha=0.1):
    score = []
    for yt, l, u in zip(y_true, lower, upper):
        if np.isnan(l) or np.isnan(u) or np.isnan(yt):
            continue
        if np.isinf(l) or np.isinf(u) or np.isinf(yt):
            continue
        if l <= yt <= u:
            score.append(u - l)
        else:
            penalty = (2 / alpha) * (l - yt) if yt < l else (2 / alpha) * (yt - u)
            score.append((u - l) + penalty)
    return np.mean(score) if score else float('nan')

# 📘 Cell 5: K-Fold Cross-Validation ile eğitim ve değerlendirme
kf = KFold(n_splits=5, shuffle=True, random_state=42)

val_scores = []
test_preds_lower = []
test_preds_upper = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    def train_qr_model(alpha):
        model = lgb.LGBMRegressor(
            objective='quantile',
            alpha=alpha,
            learning_rate=0.05,
            n_estimators=400,
            min_child_samples=20,
            max_depth=7,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50)]
        )
        return model

    model_lower = train_qr_model(0.1)
    model_upper = train_qr_model(0.9)

    pred_lower_val = safe_expm1(model_lower.predict(X_val))
    pred_upper_val = safe_expm1(model_upper.predict(X_val))
    y_val_exp = np.expm1(y_val)

    fold_score = winkler_score(y_val_exp, pred_lower_val, pred_upper_val)
    print(f"Fold {fold+1} Winkler Score: {fold_score}")
    val_scores.append(fold_score)

    test_pred_lower = safe_expm1(model_lower.predict(test_X))
    test_pred_upper = safe_expm1(model_upper.predict(test_X))

    test_preds_lower.append(test_pred_lower)
    test_preds_upper.append(test_pred_upper)

print(f"K-Fold CV Validation Winkler Score: {np.mean(val_scores)} ± {np.std(val_scores)}")

# Ortalama test tahminleri
final_pi_lower = np.mean(test_preds_lower, axis=0)
final_pi_upper = np.mean(test_preds_upper, axis=0)

# 📘 Cell 6: Submission dosyası oluştur
submission = pd.DataFrame({
    "id": test[ID_col],
    "pi_lower": final_pi_lower,
    "pi_upper": final_pi_upper
})

submission.to_csv("submission.csv", index=False)
print("✅ Yeni submission dosyası oluşturuldu.")
