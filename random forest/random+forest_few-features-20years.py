import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rc('font',size=6)
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer,mean_squared_error,mean_absolute_error
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc

def print_evaluate(true, predicted):#性能评估指标（metrics）
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)#均方误差
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')


def cal(y_true, y_pred):
    # confusion matrix row means GT, column means predication
    name = 'confusion matrix'
    '''画混淆矩阵'''
    mat = confusion_matrix(y_true, y_pred)
    da = pd.DataFrame(mat, index=['-1', '1'])
    sns.heatmap(da, annot=True, cbar=None, cmap='Blues')
    plt.title(name)
    # plt.tight_layout()yt
    plt.ylabel('True Label')
    plt.xlabel('Predict Label')

    tp = np.diagonal(mat)  # tp
    gt_num = np.sum(mat, axis=1)  # axis = 1
    pre_num = np.sum(mat, axis=0)
    fp = pre_num - tp
    fn = gt_num - tp
    num = np.sum(gt_num)
    num = np.repeat(num, gt_num.shape[0])
    gt_num0 = num - gt_num
    tn = gt_num0 - fp
    recall = tp.astype(np.float32) / gt_num
    specificity = tn.astype(np.float32) / gt_num0
    precision = tp.astype(np.float32) / pre_num
    F1 = 2 * (precision * recall) / (precision + recall)
    acc = (tp + tn).astype(np.float32) / num

    print('recall:', recall, '\nmean recall:{:.4f}'.format(np.mean(recall)))
    print('specificity:', specificity, '\nmean specificity:{:.4f}'.format(np.mean(specificity)))
    print('precision:', precision, '\nmean precision:{:.4f}'.format(np.mean(precision)))
    print('F1:', F1, '\nmean F1:{:.4f}'.format(np.mean(F1)))
    print('acc:', acc, '\nmean acc:{:.4f}'.format(np.mean(acc)))

# data set, set features and target
data=pd.read_csv('Apple_data_20years.csv')
features= data[['Open','High','Low','Volume']]

target=data['Close'];target=target.astype(float)
target=data['Close']

# rise or drop
data['change'] = data['Close']-data['Open']
data['up']= data['change']
data['up'][data['change']>=0] = 1 #rise
data['up'][data['change']<0] = -1 #drop
target1 = data['up']

#数据归一化处理
min_max_scaler = preprocessing.MinMaxScaler()#X_scaled = (X - X_min) / (X_max - X_min)
features = min_max_scaler.fit_transform(features)#两列数据转化

#数据集划分
split_num=int(len(features)*0.8)
X_train=features[:split_num]
Y_train=target[:split_num]
X_test=features[split_num:]
Y_test=target[split_num:]

Y_train1=target1[:split_num]
Y_test1=target1[split_num:]

#2. price prediction
rfr=RandomForestRegressor(n_estimators=100,random_state=42)
rfr.fit(X_train,Y_train)

train_pred = rfr.predict(X_train)
test_pred=rfr.predict(X_test)

print(test_pred)
print('price prediction:Test set evaluation:\n_____________________________________')
print_evaluate(Y_test, test_pred)
print('price prediction:Train set evaluation:\n_____________________________________')
print_evaluate(Y_train, train_pred)

#可视化
sns.set(font_scale=1.2)
plt.plot(list(range(0,len(X_test))),Y_test,marker='o',markersize=2)
plt.plot(list(range(0,len(X_test))),test_pred,marker='*',markersize=2)
plt.legend(['True','Prediction'])
plt.title('stock price prediction')
plt.show()


#rise or fall
rfc=RandomForestClassifier(n_estimators=100,random_state=42)
rfc.fit(X_train,Y_train1)

train1_pred = rfc.predict(X_train)
test1_pred=rfc.predict(X_test)

print('rise or fall:Test set evaluation:\n_____________________________________')
print_evaluate(Y_test1, test1_pred)
print('rise or fall:Train set evaluation:\n_____________________________________')
print_evaluate(Y_train1, train1_pred)

# 评估模型效果：混淆矩阵
cal(Y_test1, test1_pred)

# 评估模型效果：ROC曲线和AUC值
y_probabilities = rfc.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(Y_test1, y_probabilities)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()