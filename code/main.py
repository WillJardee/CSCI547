from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree as Tree
import RuleExtractor
import numpy as np
import TreeRuleExtractor as tRule


class Dataset:
    def __init__(self, dataset):
        dat = np.genfromtxt("../datasets/" + dataset + "/data.csv", delimiter=",", dtype=str)
        self.X, self.y = dat[:, 0:-1], dat[:, -1]
        self.xenc = OneHotEncoder(handle_unknown='ignore')
        self.x_hot = self.xenc.fit_transform(self.X)
        self.yenc= OneHotEncoder(handle_unknown='ignore')
        self.yenc.fit(np.array(np.unique(self.y)).reshape(1, -1))
        self.classes = np.unique(self.y)
        self.forest = None
        self.build_forest()

    def build_forest(self):
        f_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.forest = f_classifier.fit(self.x_hot, self.y)



# def build_dataset(dataset):
#     dat = np.genfromtxt("../datasets/" + dataset + "/data.csv", delimiter=",", dtype=str)
#     X, y = dat[:, 0:-1], dat[:, -1]
#     enc = OneHotEncoder(handle_unknown='ignore')
#     x_hot = enc.fit_transform(X)
#     ency = OneHotEncoder(handle_unknown='ignore')
#     ency.fit(np.array(np.unique(y)).reshape(1,-1))
#
#     rf = RandomForestClassifier(n_estimators=100, random_state=42)
#     return rf.fit(x_hot, y), ency, np.unique(y)


if __name__ == '__main__':
    dataset = Dataset('car')
    rf, ency, classes = dataset.forest, dataset.yenc, dataset.classes
    # tRule.tree_to_code(rf[0], [str(i) + "." for i in list(range(27))])
    # rules = tRule.get_rules(rf[0], [str(i) for i in list(range(27))], ['unacc', 'acc', 'good', 'vgood'])
    # for r in rules:
    #     print(r)
    vector = []
    for each_tree in range(len(rf)):
        rule = tRule.tree_to_code2(rf[each_tree])
        for each_rule in range(len(rule)):
            arr = -1*np.ones(21)
            for i in rule[each_rule][:-1]:
                arr[i[0]-1] = i[1]
            temp = rule[each_rule][-1]
            for i in range(len(temp[0])):
                if temp[0][i] != 0:
                    temp[0][i] = 1
                else:
                    temp[0][i] = -1
            arr = np.append(arr, [temp])
            vector.append(arr)

    ruleMap = RuleExtractor.LorentzMap(21, classes.shape[0])

    for each_vector in range(len(vector)):
        ruleMap.add_term(vector[each_vector])

    print("end map gen")
    for i in ruleMap.eigs(): print(i)
    print("end all")
