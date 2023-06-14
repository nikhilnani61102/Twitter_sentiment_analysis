import numpy as np
import matplotlib.pyplot as plt
   
Accuracy = [67.01, 68.02, 80.06]
Precision = [66.5, 69.05, 79.12]
Recall = [66.20, 68.10, 75.24]
FScore = [65.90, 68.07, 72.14]


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
plt.xticks(r + width/2,['KNN','SVM','TF-IDF'])
plt.legend()
  
plt.show()