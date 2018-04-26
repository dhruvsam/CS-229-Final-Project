
y = model.predixt(X_test)

for i in range(len(y)):
    if y[i,0] >= y[i,1]:
        y[i,0] = 1
        y[i,1] = 0
    else:
        y[i,0] = 0
        y[i,1] = 1
from sklearn.metrics import classification_report
target_names = ['class 0', 'class 1']

print(classification_report(Y_test, Y_pred, target_names=target_names))
