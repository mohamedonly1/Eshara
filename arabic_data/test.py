import numpy as np

# متوسط أول samples (بتوعك)
mean_local = kp.iloc[:200,1:].mean()

# متوسط باقي الداتا
mean_online = kp.iloc[200:,1:].mean()

diff = np.linalg.norm(mean_local - mean_online)

print(diff)