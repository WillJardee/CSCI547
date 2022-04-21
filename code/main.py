from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree as Tree
import numpy as np
import TreeRuleExtractor as tRule


def build_dataset(dataset):
    dat = np.genfromtxt("../datasets/" + dataset + "/data.csv", delimiter=",", dtype=str)
    X, y = dat[:, 0:-1], dat[:, -1]
    enc = OneHotEncoder(handle_unknown='ignore')
    x_hot = enc.fit_transform(X)
    ency = OneHotEncoder(handle_unknown='ignore')
    ency.fit(np.array(['unacc', 'acc', 'good', 'vgood']).reshape(1,-1))

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    return rf.fit(x_hot, y), ency


if __name__ == '__main__':
    rf, ency = build_dataset('car')
    # tRule.tree_to_code(rf[0], [str(i) + "." for i in list(range(27))])
    # rules = tRule.get_rules(rf[0], [str(i) for i in list(range(27))], ['unacc', 'acc', 'good', 'vgood'])
    # for r in rules:
    #     print(r)
    rule = tRule.tree_to_code2(rf[0])
    arr = -1*np.ones(21)
    for i in rule[0][:-1]:
        arr[i[0]-1] = i[1]

    temp = rule[0][-1]
    for i in range(len(temp[0])):
        if temp[0][i] != 0:
            temp[0][i] = 1
    arr = np.append(arr, [temp])
    print(arr)
    print(arr.shape)
