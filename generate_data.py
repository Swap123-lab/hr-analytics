"""Generate a synthetic but realistic HR dataset (3,000 employees).

Attrition is simulated from a logistic process driven by overtime, pay vs. job
level, promotion stagnation, satisfaction, tenure and commute, so the analysis
has real signal to find. Fixed seed => reproducible.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 3000

dept = rng.choice(["Sales", "Technology", "HR", "Finance", "Operations"], N,
                  p=[0.28, 0.32, 0.08, 0.12, 0.20])
roles = {
    "Sales": ["Sales Executive", "Account Manager", "Sales Lead"],
    "Technology": ["Developer", "Data Analyst", "QA Engineer", "Tech Lead"],
    "HR": ["HR Executive", "Recruiter", "HRBP"],
    "Finance": ["Financial Analyst", "Accountant", "Finance Manager"],
    "Operations": ["Ops Associate", "Process Lead", "Supply Planner"],
}
role = np.array([rng.choice(roles[d]) for d in dept])

age = np.clip(rng.normal(34, 8, N), 21, 60).astype(int)
level = np.clip(np.round((age - 20) / 9 + rng.normal(0, 0.8, N)), 1, 5).astype(int)
years_company = np.clip(((age - 21) * rng.beta(1.5, 3.5, N)), 0, 30).round().astype(int)
base_pay = {1: 28000, 2: 45000, 3: 70000, 4: 105000, 5: 150000}
income = np.array([base_pay[l] for l in level]) * rng.normal(1.0, 0.18, N)
income = income.round(-2).astype(int)

since_promo = np.minimum(rng.integers(0, 9, N), years_company)
overtime = rng.choice(["Yes", "No"], N, p=[0.28, 0.72])
travel = rng.choice(["None", "Rare", "Frequent"], N, p=[0.15, 0.65, 0.20])
distance = np.clip(rng.gamma(2.0, 5.0, N), 1, 45).round().astype(int)
job_sat = rng.choice([1, 2, 3, 4], N, p=[0.12, 0.20, 0.38, 0.30])
wlb = rng.choice([1, 2, 3, 4], N, p=[0.08, 0.22, 0.50, 0.20])
perf = rng.choice([1, 2, 3, 4, 5], N, p=[0.03, 0.10, 0.52, 0.27, 0.08])
training = np.clip(rng.normal(32, 14, N), 0, 90).round().astype(int)
gender = rng.choice(["Female", "Male", "Other"], N, p=[0.44, 0.54, 0.02])

df = pd.DataFrame({
    "EmployeeID": np.arange(1001, 1001 + N), "Age": age, "Gender": gender,
    "Department": dept, "JobRole": role, "JobLevel": level,
    "MonthlyIncome": income, "YearsAtCompany": years_company,
    "YearsSinceLastPromotion": since_promo, "OverTime": overtime,
    "BusinessTravel": travel, "DistanceFromHome": distance,
    "JobSatisfaction": job_sat, "WorkLifeBalance": wlb,
    "PerformanceRating": perf, "TrainingHoursLastYear": training,
})

# Pay relative to the median for the same job level (compa-ratio)
df["CompaRatio"] = df["MonthlyIncome"] / df.groupby("JobLevel")["MonthlyIncome"].transform("median")

logit = (
    -2.1
    + 1.1 * (df.OverTime == "Yes")
    + 0.5 * (df.BusinessTravel == "Frequent")
    - 2.2 * (df.CompaRatio - 1)
    + 0.22 * np.minimum(df.YearsSinceLastPromotion, 6)
    - 0.40 * (df.JobSatisfaction - 2.5)
    - 0.30 * (df.WorkLifeBalance - 2.5)
    + 0.025 * df.DistanceFromHome
    - 0.06 * df.YearsAtCompany
    + 0.35 * (df.YearsAtCompany < 2)
    - 0.008 * df.TrainingHoursLastYear
    + 0.25 * (df.Department == "Sales")
)
p = 1 / (1 + np.exp(-logit))
df["Attrition"] = np.where(rng.random(N) < p, "Yes", "No")

# Inject a little messiness so the cleaning step is real
df.loc[rng.choice(N, 60, replace=False), "JobSatisfaction"] = np.nan
df.loc[rng.choice(N, 40, replace=False), "TrainingHoursLastYear"] = np.nan

df.to_csv("data/hr_data.csv", index=False)
print(f"Saved data/hr_data.csv  rows={len(df)}  attrition={df.Attrition.eq('Yes').mean():.1%}")
