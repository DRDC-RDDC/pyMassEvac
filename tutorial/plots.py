import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

v_bar = pd.read_csv('v_bar.csv', header = None)
expected_objective = pd.read_csv('expected_objective.csv')

x = np.linspace(0, 90, 10)
sns.relplot(x = x, y = v_bar[0])
plt.show()


