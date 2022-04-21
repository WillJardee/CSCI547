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
    ency.fit(['unacc', 'acc', 'good', 'vgood'])

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    return rf.fit(x_hot, y), ency


if __name__ == '__main__':
    rf, ency = build_dataset('car')
    # tRule.tree_to_code(rf[0], [str(i) + "." for i in list(range(27))])
    # rules = tRule.get_rules(rf[0], [str(i) for i in list(range(27))], ['unacc', 'acc', 'good', 'vgood'])
    # for r in rules:
    #     print(r)
    rule = tRule.tree_to_code2(rf[50])
    arr = np.zeros(21)
    print(rule[10])
    for i in rule[10][:-1]:
        arr[i[0]-1] = i[1]
    print(arr)
    print(rf[50].predict([arr]))
    arr = np.concatenate((arr, ency.transform(rf[50].predict([arr]).reshape(1, -1))))
    print(arr)
