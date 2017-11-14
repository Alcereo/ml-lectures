import numpy as np
import matplotlib as mpl
# mpl.use("Agg")  # Must come after importing mpl, but before importing plt
import matplotlib.pyplot as plt

X_1 = np.array([[np.random.randint(10,50), np.random.randint(50,80)] for n in range(50)])

plt.figure();
plt.plot(X_1[:,0], X_1[:,1],".");
plt.show();