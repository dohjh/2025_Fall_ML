import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, 
    WhiteKernel, 
    ExpSineSquared
)

FILE_PATH = 'co2_mm_mlo.csv'

data = pd.read_csv(FILE_PATH, skiprows=40)

X_all = data['decimal date'].values
y_all = data['average'].values

train_mask = X_all < 2016.0
X_train = X_all[train_mask]
y_train = y_all[train_mask]

test_mask = (X_all >= 2016.0) & (X_all < 2026.0)
X_test = X_all[test_mask]
y_test = y_all[test_mask]

X_train_gp_input = X_train.reshape(-1, 1)
X_test_gp_input = X_test.reshape(-1, 1)


# 1. GP 커널 설계

kernel = (
    #(1) 매우 긴 장기 추세 (수십~100년)
    RBF(length_scale=50, length_scale_bounds=(40, 100))
    # (2) 계절성 (1년)
    + ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")
    # (3) 노이즈
    + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-5, 1e1))
)

# 2. GP: 모델 초기화 및 학습
gp = GaussianProcessRegressor(
    kernel=kernel, 
    n_restarts_optimizer=10,
    normalize_y=True,
    random_state=42
)

print("Starting GP Fitting (this may take a minute or two...)")
gp.fit(X_train_gp_input, y_train)
print("GP Fitting complete.")
print(f"Optimized kernel: {gp.kernel_}")


# 3. GP: 예측 (Predictive Distribution) (이전과 동일)
predictive_mean_gp, predictive_std_gp = gp.predict(X_test_gp_input, return_std=True)
ci_95_upper_gp = predictive_mean_gp + 1.96 * predictive_std_gp
ci_95_lower_gp = predictive_mean_gp - 1.96 * predictive_std_gp

# 4. GP: 결과 시각화 (이전과 동일)
plt.figure(figsize=(14, 8))
plt.plot(X_train, y_train, 'k.', markersize=2, alpha=0.1, label='Training Data (1958-2015)')
plt.plot(X_test, y_test, 'r.', markersize=5, label='Actual Test Data (2016-2025)')
plt.plot(X_test, predictive_mean_gp, 'g-', label='GP Predictive Mean (RBF+Per Kernel)')
plt.fill_between(X_test, ci_95_lower_gp, ci_95_upper_gp, color='green', alpha=0.2, label='95% Confidence Interval')
plt.title('Gaussian Process (GP): CO2 Prediction', fontsize=16)
plt.xlabel('Year (decimal date)', fontsize=12)
plt.ylabel('CO2 Concentration (ppm)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(2010, 2026)
plt.show()

# 5. GP: 수치적인 성능평가 (이전과 동일)
errors_gp = predictive_mean_gp - y_test
rmse_gp = np.sqrt(np.mean(errors_gp**2))
within_band_gp = (y_test >= ci_95_lower_gp) & (y_test <= ci_95_upper_gp)
coverage_percentage_gp = np.mean(within_band_gp) * 100
avg_predictive_std_gp = np.mean(predictive_std_gp)

print("-" * 30)
print("[GP Numerical Evaluation Results (RBF Kernel)]")
print(f"RMSE (Test Data): {rmse_gp:.4f} ppm")
print(f"Coverage (within 95% CI): {coverage_percentage_gp:.2f} %")
print(f"Average Predictive Std Dev: {avg_predictive_std_gp:.4f} ppm")
print("-" * 30)