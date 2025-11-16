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

# 1. 데이터 로드 (전체 데이터를 학습에 사용)
data = pd.read_csv(FILE_PATH, skiprows=40)

X_all = data['decimal date'].values
y_all = data['average'].values

# [Problem 2] X_all을 GP 입력으로 변환
X_all_gp_input = X_all.reshape(-1, 1)


# 1. GP: 커널 설계 (Problem 1과 동일한 성공적인 커널 사용)
kernel = (
    RBF(length_scale=50, length_scale_bounds=(40, 100))
    + ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")
    + WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-5, 1e1))
)

# 2. GP: 모델 초기화 및 학습 (전체 데이터로 학습)
gp = GaussianProcessRegressor(
    kernel=kernel, 
    n_restarts_optimizer=10,
    normalize_y=True,
    random_state=42
)

print("Starting GP Fitting (this may take a minute or two...)")
# [Problem 2] X_all, y_all로 학습
gp.fit(X_all_gp_input, y_all)
print("GP Fitting complete.")
print(f"Optimized kernel: {gp.kernel_}")


# 3. GP: 2040년까지의 미래 예측
# [Problem 2] 1980년부터 2040년까지의 예측 축 생성
X_future = np.linspace(1980, 2040, 500)
X_future_gp_input = X_future.reshape(-1, 1)

# [Problem 2] X_future로 예측
predictive_mean_gp, predictive_std_gp = gp.predict(X_future_gp_input, return_std=True)
ci_95_upper_gp = predictive_mean_gp + 1.96 * predictive_std_gp
ci_95_lower_gp = predictive_mean_gp - 1.96 * predictive_std_gp

# 4. GP: 결과 시각화 (2040년까지)
plt.figure(figsize=(14, 8))

# [Problem 2] 훈련 데이터를 X_all, y_all로 변경
plt.plot(X_all, y_all, 'r.', markersize=2, alpha=0.8, label='All Training Data (1958-2025)')
# (X_test, y_test는 더 이상 필요 없음)

# [Problem 2] 예측 축을 X_future로 변경
plt.plot(X_future, predictive_mean_gp, 'g-', label='GP Predictive Mean (RBF+Per Kernel)')
plt.fill_between(X_future, ci_95_lower_gp, ci_95_upper_gp, color='green', alpha=0.2, label='95% Confidence Interval')

plt.axvline(x=2026.0, color='gray', linestyle='--', linewidth=1.5, label='Forecast Start (2026)')


plt.title('GP: CO2 Long-term Forecast to 2040', fontsize=16)
plt.xlabel('Year (decimal date)', fontsize=12)
plt.ylabel('CO2 Concentration (ppm)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# [Problem 2] X축 범위를 1980-2040으로 변경
plt.xlim(2010, 2040) 

plt.show()

# [Problem 2] 5번 수치 평가는 y_test가 없으므로 제거 (미래 예측이므로 정답이 없음)
print("\n--- GP Problem 2 (Forecast) Complete ---")



# -----------------------------------------------
# 5. 수치적 분석: 불확실성 증가율 계산
# -----------------------------------------------

# 2026년 (외삽 시작점) 찾기
X_start_forecast = np.array([[2026.0]])
_, std_start = gp.predict(X_start_forecast, return_std=True)
std_start = std_start[0]

# 2040년 (외삽 종료점) 찾기
X_end_forecast = np.array([[2040.0]])
_, std_end = gp.predict(X_end_forecast, return_std=True)
std_end = std_end[0]

std_growth_GP = std_end - std_start
std_growth_rate_GP = (std_growth_GP / std_start) * 100

print("-" * 30)
print("[GP Uncertainty Growth Analysis]")
print(f"Std Dev @ 2026.0: {std_start:.4f} ppm")
print(f"Std Dev @ 2040.0: {std_end:.4f} ppm")
print(f"Growth (2026 -> 2040): {std_growth_GP:.4f} ppm")
print(f"Growth Rate: {std_growth_rate_GP:.2f} % (in 14 years)")
print("-" * 30)