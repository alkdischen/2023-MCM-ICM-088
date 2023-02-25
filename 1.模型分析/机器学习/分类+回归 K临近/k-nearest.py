# #two-class dataset for classification.
# import mglearn
# import matplotlib.pyplot as plt
# X,y=mglearn.datasets.make_forge()
# mglearn.discrete_scatter(X[:,0],X[:,1],y)
# plt.legend(["Class 0","Class 1"],loc=4)
# plt.xlabel("First feature")
# plt.ylabel("Second feature")
# print("X.shape:{}".format(X.shape))
# # plt.show()
#
# #wave dataset is used for regression
# X1,y1=mglearn.datasets.make_wave(n_samples=40)
# plt.plot(X1,y1,'o')
# plt.ylim(-3,3)
# plt.xlabel("Feature")
# plt.ylabel("Target")
# # plt.show()
#
# #the cancer dataset(classification)
# from sklearn.datasets import load_breast_cancer
# import numpy as np
# cancer=load_breast_cancer()
# print("cancer.keys: {}".format(cancer.keys()))
# #datasets in sklearn are stored as Bunch objects
# print("Sample counts per class:\n{}".format(
#     {n:v for n,v in zip(cancer.target_names,np.bincount(cancer.target))}
# ))
# print("Feature names:\n{}".format(cancer.feature_names))
#
# #the california dataset(regression)
# from sklearn.datasets import fetch_california_housing
# cali=fetch_california_housing()
# print("Data shape:{}".format(cali.data.shape))

#第一块
import mglearn.datasets
from sklearn.model_selection import train_test_split
X,y=mglearn.datasets.make_forge()
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
clf=KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train,y_train)
print("Test set accuracy: {:.2f}".format(clf.score(X_test,y_test)))
#The following code produces the visualization of the
#decision boundaries for one, three, and nine neighbors.
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(10,3))
for n_neighbors,ax in zip([1,3,9],axes):
    #the fit method returns the obj self, so we can instantiate
    #and fit in one line
    clf=KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf,X,fill=True,eps=0.5,ax=ax,alpha=.4)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("First feature")
    ax.set_ylabel("Second feature")
axes[0].legend(loc=3)
plt.show()
#第二块
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() #下载数据
X_train, X_test, y_train, y_test = train_test_split(
 cancer.data, cancer.target, stratify=cancer.target, random_state=66) #分开数据
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)#对于k=1-10的情况逐个讨论
    clf.fit(X_train, y_train)#训练
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))#把各个k的相对于training set和test set的准确率塞进list中
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))
#后面这个就不太懂了，大概是按照数据描折线图以及确定横纵坐标？
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()
#第三块
mglearn.plots.plot_knn_regression(n_neighbors=3)#k=3的回归模型
plt.show()
#第四块
#用K临近做回归其实就是取靠近test数据自变量值的k个因变量的平均值
from sklearn.neighbors import KNeighborsRegressor
X,y=mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)#下载、分数据
reg=KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train,y_train)
print("Test set predictions:\n{}".format(reg.score(X_test,y_test)))#建立对象、训练模型、输出评估结果
#这个matplotlib请另一位同学为我解释一下吧...
import numpy as np
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# create 1,000 data points, evenly spaced between -3 and 3
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
     # make predictions using 1, 3, or 9 neighbors
     reg = KNeighborsRegressor(n_neighbors=n_neighbors)
     reg.fit(X_train, y_train)
     ax.plot(line, reg.predict(line))
     ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
     ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
     ax.set_title(
     "{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
     n_neighbors, reg.score(X_train, y_train),
     reg.score(X_test, y_test)))
     ax.set_xlabel("Feature")
     ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target",
 "Test data/target"], loc="best")
plt.show()