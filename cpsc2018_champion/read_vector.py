import numpy as np

# 讀取 .npy 檔案
file_path = r'cpsc2018_champion\magicVector_test_val_strategy.npy'
data = np.load(file_path)

# 顯示讀取的數據
print(data)
