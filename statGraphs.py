import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid", palette="muted")
plt.figure(figsize=(140,140))

def load_dataset(path):
    data = pd.read_csv(path)
    return data

def dependent_variable_trend(data, parameter, parameter_label):
    index = np.array([i for i in range(1, 7)])
    close_param = np.sort(np.array(data.loc[data['distance'] == "close"][parameter]))
    far_param = np.sort(np.array(data.loc[data['distance'] == "far"][parameter]))
    plt.plot(index, close_param,c='b',marker="^",ls='--',label= parameter_label + ' for close range',fillstyle='none')
    plt.plot(index, far_param,c='g',marker='v',ls='--',label= parameter_label +' for far range',fillstyle='none')
    plt.xlabel('Users in sorted based on response time for each (close or far) range type', fontsize=10)
    plt.ylabel('User ' + parameter_label, fontsize=10)
    plt.legend(loc=2)
    plt.show()

path = "dataset2.csv"
data = load_dataset(path)

dependent_variable_trend(data, "time", "Response Time (seconds)")
dependent_variable_trend(data, "error", "Error Rate  (percentage)")