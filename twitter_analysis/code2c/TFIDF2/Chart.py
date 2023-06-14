import numpy as np
import matplotlib.pyplot as plt
   
Accuracy = [97.84, 97.3, 96.68, 97.66, 98.38, 99.06]
Precision = [98.2, 97.88, 98.17, 98.19, 98.31, 99.09]
Recall = [98.25, 98.93, 97.86, 99.04, 98.89, 99.04]
FScore = [98.72, 98.40, 98.01, 98.61, 99.04, 99.06]


n=6
r = np.arange(n)
width = 0.20
  
  
plt.bar(r, Accuracy, color = 'b',
        width = width, edgecolor = 'black',
        label='Accuracy')
plt.bar(r + width, Precision, color = 'g',
        width = width, edgecolor = 'black',
        label='Precision')
plt.bar(r + width + 0.20, Recall, color = 'r',
        width = width, edgecolor = 'black',
        label='Recall')
plt.bar(r + width + 0.40, FScore, color = 'y',
        width = width, edgecolor = 'black',
        label='FScore')


  
plt.xlabel("Comparision Algorithms")
plt.ylabel("Peformance Value(%)")
plt.title("Performance Comparision")
  
# plt.grid(linestyle='--')
plt.xticks(r + width/2,['1_CNN','2_CNN','3_CNN','LSTM','DCF','IDCF'])
plt.legend()
  
plt.show()