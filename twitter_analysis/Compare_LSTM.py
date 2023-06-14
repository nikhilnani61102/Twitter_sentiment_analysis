import numpy as np
import matplotlib.pyplot as plt
   
Accuracy = [85.54, 68.02, 80.06]
Precision = [80.15, 69.05, 79.12]
Recall = [80.24, 68.10, 75.24]
FScore = [81.14, 68.07, 72.14]


n=3
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
plt.xticks(r + width/2,['LSTM','SVM','Novel TF-IDF'])
plt.legend()
  
plt.show()