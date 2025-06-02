import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
import lightgbm as lgb
import matplotlib.pyplot as plt

# Veri yükle
train = pd.read_csv("house-pi/dataset/dataset.csv")
test = pd.read_csv("house-pi/test.csv")
target_col = "sale_price"
ID_col = "id"

# Outlier filtreleme
train = train[train[target_col] < 1_500_000]

# Tarih dönüşümü
train["sale_date"] = pd.to_datetime(train["sale_date"])
test["sale_date"] = pd.to_datetime(test["sale_date"])
train["sale_year"] = train["sale_date"].dt.year
train["sale_month"] = train["sale_date"].dt.month
test["sale_year"] = test["sale_date"].dt.year
test["sale_month"] = test["sale_date"].dt.month

# Yeni featurelar
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
num_cols = [c for c in num_cols if c not in [target_col, ID_col]]

# Encode categorical
for col in categorical_cols:
    train[col] = train[col].astype(str).fillna("missing")
    test[col] = test[col].astype(str).fillna("missing")

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train[categorical_cols] = encoder.fit_transform(train[categorical_cols])
test[categorical_cols] = encoder.transform(test[categorical_cols])

full_features = list(dict.fromkeys(num_cols + categorical_cols))

train_clean = train.dropna(subset=full_features + [target_col])
X = train_clean[full_features]
y = np.log1p(train_clean[target_col])
test_X = test[full_features]

# Winkler skor fonksiyonu (nan/inf korumalı)
def winkler_score(y_true, lower, upper, alpha=0.1):
    scores = []
    for yt, l, u in zip(y_true, lower, upper):
        if np.isnan(l) or np.isnan(u) or np.isnan(yt):
            continue
        if np.isinf(l) or np.isinf(u) or np.isinf(yt):
            continue
        if l <= yt <= u:
            scores.append(u - l)
        else:
            penalty = (2 / alpha) * (l - yt) if yt < l else (2 / alpha) * (yt - u)
            scores.append((u - l) + penalty)
    return np.mean(scores) if scores else float('nan')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

preds_lower = np.zeros(len(test_X))
preds_upper = np.zeros(len(test_X))
val_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Model alt ve üst sınır için
    model_lower = lgb.LGBMRegressor(
        objective='quantile', alpha=0.05,
        learning_rate=0.03, n_estimators=2000,
        min_child_samples=40, max_depth=5,
        subsample=0.7, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=2.0,
        random_state=42
    )
    model_upper = lgb.LGBMRegressor(
        objective='quantile', alpha=0.95,
        learning_rate=0.03, n_estimators=2000,
        min_child_samples=40, max_depth=5,
        subsample=0.7, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=2.0,
        random_state=42
    )
    
    model_lower.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50)],
    )
    model_upper.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50)],
    )
    
    pred_lower_val = np.expm1(model_lower.predict(X_val))
    pred_upper_val = np.expm1(model_upper.predict(X_val))
    y_val_exp = np.expm1(y_val)
    val_score = winkler_score(y_val_exp, pred_lower_val, pred_upper_val)
    print(f"Fold {fold + 1} Winkler Score: {val_score}")
    val_scores.append(val_score)
    
    preds_lower += np.expm1(model_lower.predict(test_X)) / kf.n_splits
    preds_upper += np.expm1(model_upper.predict(test_X)) / kf.n_splits

print(f"K-Fold CV Validation Winkler Score: {np.mean(val_scores)} ± {np.std(val_scores)}")

# Feature importance için son foldun modelini kullanalım
lgb.plot_importance(model_upper, max_num_features=20)
plt.title("Feature importance (upper bound model)")
plt.show()

# Submission dosyası
submission = pd.DataFrame({
    "id": test[ID_col],
    "pi_lower": preds_lower,
    "pi_upper": preds_upper
})
submission.to_csv("submission.csv", index=False)
print("✅ Yeni submission dosyası oluşturuldu.")
