import matplotlib.pyplot as plt
import numpy as np
import cv2

labels = ['5ft', '8ft', '11ft']
predicted_distances = [59.96, 96.45, 121.83]  # Example predicted distances in inches
actual_distances = [60, 96, 132]  # Actual distances in inches

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, predicted_distances, width, label='Predicted Distance', color='skyblue')
rects2 = ax.bar(x + width/2, actual_distances, width, label='Actual Distance', color='lightgreen')
ax.set_ylabel('Distance (inches)')
ax.set_title('Predicted vs Actual Distances')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.savefig('distance_comparison.png')
plt.show()
