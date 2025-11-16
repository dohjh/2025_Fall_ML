import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 'linear': 1차 선형 함수 (M=2) 
# 'quadratic': 2차 다항 함수 (M=3) 
# 'full_7D': 2차 다항 + 주기 함수 (M=7)
BASIS_FUNCTION_MODE = 'full_7D' 
# -----------------------------------------------

print(f"--- Running BLR with mode: {BASIS_FUNCTION_MODE} ---")

# 1. 데이터 로드 및 분할
FILE_PATH = 'co2_mm_mlo.csv'
data = pd.read_csv(FILE_PATH, skiprows=40)

X_all = data['decimal date'].values
y_all = data['average'].values

# 1958-2015 (Train data)
# 2016-2025 (Test data)
train_mask = X_all < 2016.0
X_train = X_all[train_mask]
y_train = y_all[train_mask]

test_mask = (X_all >= 2016.0) & (X_all < 2026.0)
X_test = X_all[test_mask]
y_test = y_all[test_mask]

# 정규화를 위한 훈련 데이터 평균 계산
train_mean = X_train.mean()

# 2. BLR 기저 함수 
def create_phi(X, mode, reference_mean):
    X_norm = X - reference_mean # 훈련 데이터의 평균을 기준으로 정규화
    N = len(X)

    if mode == 'linear':
        # M=2: [1, x]
        Phi = np.ones((N, 2)) 
        Phi[:, 1] = X_norm    
        
    elif mode == 'quadratic':
        # M=3: [1, x, x^2]
        Phi = np.ones((N, 3)) 
        Phi[:, 1] = X_norm 
        Phi[:, 2] = X_norm**2 
        
    elif mode == 'full_7D':
        # M=7: [1, x, x^2, sin(2pi*x), cos(2pi*x), sin(4pi*x), cos(4pi*x)]
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

# 3. BLR: 모델 학습: posterior 계산
sigma_w_sq = 100.0  # 사전 분산 (sigma_w^2)
sigma_n_sq = 2**2  # 노이즈 분산 (sigma^2)

Phi_train = create_phi(X_train, mode=BASIS_FUNCTION_MODE, reference_mean=train_mean)
M = Phi_train.shape[1] # 기저 함수의 개수

#Posterior 계산 
# Sigma = ( (sigma_w^-2)*I + (sigma^-2) * Phi.T @ Phi )^-1
#mu = (sigma^-2) * Sigma @ Phi.T @ y
try:
    # (sigma_w^-2) 와 (sigma^-2) (정밀도) 계산
    inv_sigma_w_sq = 1.0 / sigma_w_sq
    inv_sigma_n_sq = 1.0 / sigma_n_sq

    Sigma_N_inv = (inv_sigma_w_sq * np.eye(M)) + (inv_sigma_n_sq * Phi_train.T @ Phi_train)
    Sigma_N = np.linalg.inv(Sigma_N_inv)
    
    mu_N = inv_sigma_n_sq * (Sigma_N @ Phi_train.T @ y_train)
    
    print("\nBLR Posterior calculated successfully (using lecture note notation).")
    print(f"mu_N (mu) shape: {mu_N.shape}")
    print(f"Sigma_N (Sigma) shape: {Sigma_N.shape}")

except np.linalg.LinAlgError as e:
    print(f"\nError calculating posterior (Singular matrix): {e}")


# 4. BLR: 예측 Predictive Distribution
Phi_test = create_phi(X_test, mode=BASIS_FUNCTION_MODE, reference_mean=train_mean)

# 예측 분포 계산
#y_mean = mu_N.T @ phi_star
#y_var = sigma_n_sq + phi_star.T @ Sigma_N @ phi_star

predictive_mean = Phi_test @ mu_N
predictive_var = sigma_n_sq + np.diag(Phi_test @ Sigma_N @ Phi_test.T) 
predictive_std = np.sqrt(predictive_var)

# 95% 신뢰 구간
ci_95_upper = predictive_mean + 1.96 * predictive_std
ci_95_lower = predictive_mean - 1.96 * predictive_std


# 5. BLR: 결과 시각화 

plt.figure(figsize=(14, 8))

plt.plot(X_train, y_train, 'k.', markersize=2, alpha=0.1, label='Training Data (1958-2015)')
plt.plot(X_test, y_test, 'r.', markersize=5, label='Actual Test Data (2016-2025)')

plt.plot(X_test, predictive_mean, 'b-', label=f'BLR Predictive Mean (M={M})')
plt.fill_between(X_test, ci_95_lower, ci_95_upper, color='blue', alpha=0.2, label='95% Confidence Interval')

plt.title(f'BLR: CO2 Prediction (Basis Mode: {BASIS_FUNCTION_MODE})', fontsize=16)
plt.xlabel('Year (decimal date)', fontsize=12)
plt.ylabel('CO2 Concentration (ppm)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(2010, 2026) # X축은 확대해서 예측 부분만 보여주는 걸로

plt.show()





# 6. 수치적인 성능평가

# 1. RMSE 
errors = predictive_mean - y_test
mse = np.mean(errors**2)
rmse = np.sqrt(mse)
print("-" * 30)
print(f"[Numerical Evaluation Results (M={M})]")
print(f"RMSE (Test Data): {rmse:.4f} ppm")

# 2. 95% 신뢰 구간 (Coverage)
within_band = (y_test >= ci_95_lower) & (y_test <= ci_95_upper)
coverage_percentage = np.mean(within_band) * 100
print(f"Coverage (within 95% CI): {coverage_percentage:.2f} %")

# 3. 평균 예측 표준편차 (불확실성 크기)
avg_predictive_std = np.mean(predictive_std)
print(f"Average Predictive Std Dev: {avg_predictive_std:.4f} ppm")
print("-" * 30)