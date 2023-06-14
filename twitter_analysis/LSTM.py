import matplotlib.pyplot as plt
 
#data
x = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
h = [85.54, 80.15, 80.54, 81.14]
c = ['red', 'yellow', 'blue','pink']
 
#bar plot
plt.bar(x, height = h, color = c)
 
plt.show()