import numpy as np
import matplotlib.pyplot as plt
   
Accuracy = [84.64, 95.15, 88.23, 95.96, 98.38, 99.7]
Precision = [97.86, 96.12, 96.2, 96.44, 98.31, 99.16]
Recall = [83.49, 98.18, 89.5, 98.82, 98.09, 99.18]
FScore = [90.11, 97.14, 92.73, 97.62, 99.04, 99.17]
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
plt.xticks(r + width/2,['SVM','NB','KNN','RF','DCF','IDCF'])
plt.legend()
  
plt.show()