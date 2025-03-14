import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# ----- 1) Generate the extended dataset with 3 classes ----- #
np.random.seed(1)

# Class -1 and Class +1 (200 points)
X = np.random.randn(200, 2)
y = np.array([-1]*100 + [1]*100)
X[y == 1] += 1

# Now add 50 points for Class 0
X_new = np.random.randn(50, 2)
X_new += 1  # shift them a bit
y_new = np.array([0]*50)

# Combine old and new data
X_extended = np.vstack([X, X_new])
y_extended = np.concatenate([y, y_new])

# ----- 2) Fit an SVM on the 3-class data ----- #
C = 500
gamma = 0.01
svm_model = make_pipeline(SVC(kernel='rbf', C=C, gamma=gamma))
svm_model.fit(X_extended, y_extended)

# ----- 3) Create a mesh grid for plotting decision boundaries ----- #
x_min, x_max = X_extended[:, 0].min() - 1, X_extended[:, 0].max() + 1
y_min, y_max = X_extended[:, 1].min() - 1, X_extended[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predict class labels for each point in the grid
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# ----- 4) Map labels (-1, 0, 1) to indices (0, 1, 2) for a 3-color plot ----- #
unique_labels = np.unique(y_extended)   # e.g. array([-1,  0,  1])
label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
Z_idx = np.vectorize(label_to_idx.get)(Z)  # convert each label to 0,1,2

# ----- 5) Define a colormap for 3 classes (red, green, blue for example) ----- #
cmap = ListedColormap(['red', 'green', 'blue'])

plt.figure(figsize=(6, 5))
# Plot the 3-class decision regions
plt.contourf(xx, yy, Z_idx, alpha=0.3, cmap=cmap, levels=[-0.5, 0.5, 1.5, 2.5])

# ----- 6) Plot the training points by their true class ----- #
# We'll use the same color scheme: class -1=red, 0=green, 1=blue
for lab, color in zip(unique_labels, ['red', 'green', 'blue']):
    plt.scatter(X_extended[y_extended == lab, 0],
                X_extended[y_extended == lab, 1],
                color=color,
                label=f'Class {lab}')

# ----- 7) Plot support vectors ----- #
sv = svm_model.named_steps['svc'].support_vectors_
plt.scatter(sv[:, 0], sv[:, 1], s=100, facecolors='none',
            edgecolors='black', linewidths=1.5, label="Support Vectors")

plt.xlabel('Predictor 1')
plt.ylabel('Predictor 2')
plt.title('SVM Classification with 3 Classes')
plt.legend()
plt.show()

# ----- 8) Evaluate training errors ----- #
y_pred = svm_model.predict(X_extended)
errors = (y_pred != y_extended).sum()
print(f"Number of training errors: {errors}")
print(f"Support vector indices: {svm_model.named_steps['svc'].support_}")
print(f"Total number of support vectors: {len(svm_model.named_steps['svc'].support_)}")
