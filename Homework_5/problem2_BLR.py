import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASIS_FUNCTION_MODE = 'full_7D' 

print(f"BLR (Problem 2) with mode: {BASIS_FUNCTION_MODE} ---")

# 1. 데이터 로드 (전체 데이터를 학습에 사용)
FILE_PATH = 'co2_mm_mlo.csv'
data = pd.read_csv(FILE_PATH, skiprows=40)

X_all = data['decimal date'].values
y_all = data['average'].values

reference_mean = X_all.mean() # 문제2 전체평균을 이용해서 정규화

# 2. BLR 기저 함수
def create_phi(X, mode, reference_mean):
    X_norm = X - reference_mean 
    N = len(X)

    if mode == 'linear':
        Phi = np.ones((N, 2)) 
        Phi[:, 1] = X_norm    
    elif mode == 'quadratic':
        Phi = np.ones((N, 3)) 
        Phi[:, 1] = X_norm 
        Phi[:, 2] = X_norm**2 
    elif mode == 'full_7D':
        Phi = np.ones((N, 7)) 
        Phi[:, 1] = X_norm      
        Phi[:, 2] = X_norm**2   
        Phi[:, 3] = np.sin(2 * np.pi * X) 
        Phi[:, 4] = np.cos(2 * np.pi * X) 
        Phi[:, 5] = np.sin(4 * np.pi * X) 
        Phi[:, 6] = np.cos(4 * np.pi * X) 
    else:
        raise ValueError(f"Unknown BASIS_FUNCTION_MODE: {mode}")
    return Phi

# 3. BLR: 모델 학습(전체 데이터로 Posterior 계산)
sigma_w_sq = 100.0  # 사전 분산 (sigma_w^2)
sigma_n_sq = 2**2  # 노이즈 분산 (sigma^2) 

#  문제2 X_all, y_all로 트레이닝
Phi_all = create_phi(X_all, mode=BASIS_FUNCTION_MODE, reference_mean=reference_mean)
M = Phi_all.shape[1] 

try:
    inv_sigma_w_sq = 1.0 / sigma_w_sq
    inv_sigma_n_sq = 1.0 / sigma_n_sq

    Sigma_N_inv = (inv_sigma_w_sq * np.eye(M)) + (inv_sigma_n_sq * Phi_all.T @ Phi_all)
    Sigma_N = np.linalg.inv(Sigma_N_inv)
    
    mu_N = inv_sigma_n_sq * (Sigma_N @ Phi_all.T @ y_all)
    
    print("\nBLR Posterior calculated successfully (using ALL data).")

except np.linalg.LinAlgError as e:
    print(f"\nError calculating posterior (Singular matrix): {e}")


# 4. BLR: 2040년까지의 미래 예측
# [Problem 2] 1980년부터 2040년까지의 예측 축 생성
X_future = np.linspace(1980, 2040, 500)
Phi_future = create_phi(X_future, mode=BASIS_FUNCTION_MODE, reference_mean=reference_mean)

# 예측 분포 계산
predictive_mean = Phi_future @ mu_N
predictive_var = sigma_n_sq + np.diag(Phi_future @ Sigma_N @ Phi_future.T) 
predictive_std = np.sqrt(predictive_var)

# 95% 신뢰 구간
ci_95_upper = predictive_mean + 1.96 * predictive_std
ci_95_lower = predictive_mean - 1.96 * predictive_std


# 5. BLR: 결과 시각화 (2040년까지)

plt.figure(figsize=(14, 8))

# [Problem 2] 훈련 데이터를 X_all, y_all로 변경
plt.plot(X_all, y_all, 'r.', markersize=2, alpha=0.8, label='All Training Data (1958-2025)')


# [Problem 2] 예측 축을 X_future로 변경
plt.plot(X_future, predictive_mean, 'b-', label=f'BLR Predictive Mean (M={M})')
plt.fill_between(X_future, ci_95_lower, ci_95_upper, color='blue', alpha=0.2, label='95% Confidence Interval')

plt.axvline(x=2026.0, color='gray', linestyle='--', linewidth=1.5, label='Forecast Start (2026)')

plt.title(f'BLR: CO2 Long-term Forecast to 2040 (M={M})', fontsize=16)
plt.xlabel('Year (decimal date)', fontsize=12)
plt.ylabel('CO2 Concentration (ppm)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# [Problem 2] X축 범위를 1980-2040으로 변경
plt.xlim(2010, 2040) 

plt.show()

print("\n--- BLR Problem 2 (Forecast) Complete ---")


# -----------------------------------------------
# 6. 수치적 분석: 불확실성 증가율 계산
# -----------------------------------------------

# 2026년 (외삽 시작점) 찾기
X_start_forecast = np.array([[2026.0]])
Phi_start = create_phi(X_start_forecast, mode=BASIS_FUNCTION_MODE, reference_mean=reference_mean)
var_start = sigma_n_sq + np.diag(Phi_start @ Sigma_N @ Phi_start.T)
std_start = np.sqrt(var_start)[0]

# 2040년 (외삽 종료점) 찾기
X_end_forecast = np.array([[2040.0]])
Phi_end = create_phi(X_end_forecast, mode=BASIS_FUNCTION_MODE, reference_mean=reference_mean)
var_end = sigma_n_sq + np.diag(Phi_end @ Sigma_N @ Phi_end.T)
std_end = np.sqrt(var_end)[0]

std_growth_BLR = std_end - std_start
std_growth_rate_BLR = (std_growth_BLR / std_start) * 100

print("-" * 30)
print("[BLR Uncertainty Growth Analysis]")
print(f"Std Dev @ 2026.0: {std_start:.4f} ppm")
print(f"Std Dev @ 2040.0: {std_end:.4f} ppm")
print(f"Growth (2026 -> 2040): {std_growth_BLR:.4f} ppm")
print(f"Growth Rate: {std_growth_rate_BLR:.2f} % (in 14 years)")
print("-" * 30)