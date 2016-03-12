import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

data = np.loadtxt("data.txt", delimiter=',')
dim = data.shape

X = data[:, :dim[1]-1]
y = data[:, dim[1]-1]

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), 
	algorithm="SAMME", n_estimators=200)

bdt.fit(X, y)

# ploting parameters setting
plot_colors = "br"
plot_step = 0.2
class_names = "AB"

# plt.figure(figsize=(10, 5))

# plot the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
	np.arange(y_min, y_max, plot_step))
Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

# plot the training points
for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=c,
    	cmap=plt.cm.Paired, label="Class %s" % n)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Decision Boundary')

plt.show()