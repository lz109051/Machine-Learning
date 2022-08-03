import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_hastie_10_2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
def get_error_rate(pred,y):
    return sum(pred != y)/float(len(y))


def print_error_rate(err):
    print("Error rate: Train: %.3f - Test: %.3f" % err)


def generic_clf(X_train, Y_train, X_test, Y_test, clf):
    clf.fit(X_train,Y_train)
    pred_train = clf.predict(X_train)  #直接输出的标签
    pred_test = clf.predict(X_test)

    return get_error_rate(pred_train, Y_train),get_error_rate(pred_test, Y_test)


def adaboost_clf(X_train, Y_train, X_test, Y_test, n_estimators, clf):
    n_train,n_test = len(X_train), len(X_test)
    #初始化样本权重
    w = np.full(n_train,1/n_train)
    pred_train, pred_test = np.zeros(n_train),np.zeros(n_test)

    for i in range(n_estimators):
        clf.fit(X_train, Y_train, sample_weight=w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)
        miss = [int(x) for x in (pred_train_i != Y_train)]
        miss2 = [x if x == 1 else -1 for x in miss]

        err_m = np.dot(w, miss)/sum(w)  #误差率
        #分类器权重
        alpha_m = 0.5*np.log((1-err_m)/err_m)
        # 更新样本权重
        w = np.multiply(w,np.exp([float(x)*alpha_m for x in miss2]))  #可以看一下multipely、dot、*的区别
        #基分类器预测值求解

        pred_train = [sum(x) for x in zip(pred_train, [x*alpha_m for x in pred_train_i])]  #基分类器线性组合
        pred_test = [sum(x) for x in zip(pred_test, [x*alpha_m for x in pred_test_i])]

    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)

    return get_error_rate(pred_train,Y_train),get_error_rate(pred_test, Y_test)

def plot_error_rate(er_train, er_test):
    df_error = pd.DataFrame([er_train, er_test]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth = 3, figsize = (8,6),
            color = ['lightblue', 'darkblue'], grid = True)
    plot1.set_xlabel('Number of iterations', fontsize = 12)
    plot1.set_xticklabels(range(0,450,50))
    plot1.set_ylabel('Error rate', fontsize = 12)
    plot1.set_title('Error rate vs number of iterations', fontsize = 16)
    plt.axhline(y=er_test[0], linewidth=1, color = 'red', ls = 'dashed')


X_data, Y_data = make_hastie_10_2()

# df = pd.DataFrame(X_data)
# df["Y"] = Y_data
#
# print(df.columns)

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)


#决策树训练
clf = DecisionTreeClassifier(max_depth=3, random_state=1)
err_clf = generic_clf(X_train, y_train,X_test, y_test, clf)
print(f"DT: err_train:{err_clf[0]}, err_test:{err_clf[1]}")
err_train = []
err_test = []
x_range = range(10,410,10)

for i in x_range:
    err_ada = adaboost_clf(X_train, y_train, X_test, y_test, i, clf)
    err_train.append(err_ada[0])
    err_test.append(err_ada[1])

plt.figure("err")
plt.plot(range(10,410,10),err_train, label="err_train")
plt.plot(range(10,410,10),err_test, label = "err_test")
plt.legend()
plt.show()




