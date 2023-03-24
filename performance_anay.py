import matplotlib.pyplot as plt
import numpy as np

valid_curves = np.load('./accuracy/valid_mlp_PTC_MR.npy')
avg_valid_curve = np.mean(valid_curves, axis=0)
plt.plot(valid_curves.T, alpha=0.5)
plt.plot(avg_valid_curve)
plt.show()
print(np.max(avg_valid_curve))
