"""HR attrition analytics: EDA -> feature engineering -> models -> risk scoring."""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
FIG = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

df = pd.read_csv("data/hr_data.csv")
df["AttritionFlag"] = (df.Attrition == "Yes").astype(int)

# ---------- 1. Descriptive analytics ----------
overall = df.AttritionFlag.mean()
print(f"Overall attrition: {overall:.1%}\n")

def rate_by(col, title, fname):
    t = df.groupby(col, observed=True).AttritionFlag.agg(["mean", "size"]).rename(
        columns={"mean": "attrition_rate", "size": "headcount"})
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(t.index.astype(str), t.attrition_rate * 100, color="#2b6cb0")
    ax.axhline(overall * 100, color="#c53030", ls="--", label=f"Overall {overall:.1%}")
    ax.set_title(title); ax.set_ylabel("Attrition %"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / fname); plt.close(fig)
    return t

print(rate_by("Department", "Attrition rate by department", "attrition_by_department.png"), "\n")
print(rate_by("OverTime", "Attrition rate: overtime vs not", "attrition_by_overtime.png"), "\n")

df["TenureBand"] = pd.cut(df.YearsAtCompany, [-1, 1, 3, 5, 10, 40], labels=["0-1", "2-3", "4-5", "6-10", "10+"])
print(rate_by("TenureBand", "Attrition rate by tenure (years)", "attrition_by_tenure.png"), "\n")

df["PayBand"] = pd.cut(df.CompaRatio, [0, 0.85, 0.95, 1.05, 1.15, 5],
                       labels=["<0.85", "0.85-0.95", "0.95-1.05", "1.05-1.15", ">1.15"])
print(rate_by("PayBand", "Attrition by pay vs. level median (compa-ratio)", "attrition_by_compa_ratio.png"), "\n")

# ---------- 2. Modelling ----------
y = df.AttritionFlag
X = df.drop(columns=["EmployeeID", "Attrition", "AttritionFlag", "TenureBand", "PayBand"])
num = X.select_dtypes("number").columns.tolist()
cat = X.select_dtypes(exclude="number").columns.tolist()

def prep(scale):
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])

models = {
    "Logistic Regression": Pipeline([("prep", prep(True)), ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    "Random Forest": Pipeline([("prep", prep(False)), ("m", RandomForestClassifier(
        n_estimators=300, min_samples_leaf=10, class_weight="balanced_subsample", random_state=42, n_jobs=-1))]),
}
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
results = {}
for name, pipe in models.items():
    cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    results[name] = roc_auc_score(y_te, proba)
    print(f"{name}: CV AUC {cv_auc.mean():.3f} (+/- {cv_auc.std():.3f}) | Test AUC {results[name]:.3f}")
    RocCurveDisplay.from_predictions(y_te, proba, name=name, ax=ax)
ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_title("ROC curves (hold-out set)")
fig.tight_layout(); fig.savefig(FIG / "roc_curves.png"); plt.close(fig)

best_name = max(results, key=results.get)
best = models[best_name]
print(f"\nBest model: {best_name}")
print(classification_report(y_te, (best.predict_proba(X_te)[:, 1] >= 0.5).astype(int), target_names=["Stay", "Leave"]))

# ---------- 3. Drivers (permutation importance on raw features) ----------
imp = permutation_importance(best, X_te, y_te, scoring="roc_auc", n_repeats=15, random_state=42, n_jobs=-1)
imp_s = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)
print("Top drivers (drop in AUC when shuffled):\n", imp_s.head(8).round(4), "\n")
top = imp_s.head(10)[::-1]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.barh(top.index, top.values, color="#2b6cb0"); ax.set_title(f"Top attrition drivers ({best_name})")
ax.set_xlabel("Mean drop in ROC-AUC when feature is shuffled")
fig.tight_layout(); fig.savefig(FIG / "feature_importance.png"); plt.close(fig)

# ---------- 4. Risk scoring for current employees ----------
best.fit(X, y)
df["AttritionRisk"] = best.predict_proba(X)[:, 1]
df["RiskTier"] = pd.qcut(df.AttritionRisk, [0, 0.6, 0.9, 1.0], labels=["Low", "Medium", "High"])
active = df[df.Attrition == "No"].sort_values("AttritionRisk", ascending=False)
cols = ["EmployeeID", "Department", "JobRole", "JobLevel", "YearsAtCompany", "OverTime",
        "CompaRatio", "YearsSinceLastPromotion", "JobSatisfaction", "AttritionRisk", "RiskTier"]
active[active.RiskTier == "High"][cols].round(3).to_csv("reports/high_risk_active_employees.csv", index=False)
print("High-risk active employees flagged:", int((active.RiskTier == "High").sum()))

# ---------- 5. Business impact (assumption-driven) ----------
REPLACEMENT_COST_MONTHS = 6   # assumption: cost to replace = 6 months of pay
SAVE_RATE = 0.20              # assumption: retention actions save 20% of flagged leavers
high = active[active.RiskTier == "High"]
exp_leavers = high.AttritionRisk.sum()
cost = (high.MonthlyIncome * REPLACEMENT_COST_MONTHS * high.AttritionRisk).sum()
print(f"Expected leavers among high-risk group: {exp_leavers:.0f}")
print(f"Expected replacement cost at risk: {cost:,.0f}; potential saving at {SAVE_RATE:.0%} save rate: {cost*SAVE_RATE:,.0f}")
