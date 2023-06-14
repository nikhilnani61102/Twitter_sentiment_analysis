import numpy as np
import matplotlib.pyplot as plt
 
  
# creating the dataset
data = {'Accuracy':80.06, 'Precision':79.12, 'Recall':75.24,
        'F1-Score':72.14}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
c = ['red', 'yellow', 'blue','pink']

# creating the bar plot
plt.bar(courses, values, color =c,
        width = 0.4)
 
plt.xlabel("Performance Metrics")
plt.ylabel("Performance Value(%)")
plt.title("Novel TF-IDF Performance")
plt.show()