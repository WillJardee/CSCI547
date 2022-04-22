from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree as Tree
import RuleExtractor
import numpy as np
import TreeRuleExtractor as tRule


class Dataset:
    def __init__(self, data_name):
        self.y, self.X, self.features, self.classes = None, None, None, None
        self.xenc, self.x_hot, self.yenc = None, None, None
        self.forest = None

        self.get_dat(data_name)
        self.hot_encoding()
        self.build_forest()

    def get_dat(self, data_name):
        dat = np.genfromtxt("../datasets/" + data_name + "/data.csv", delimiter=",", dtype=str)
        self.X, self.y = dat[:, 0:-1], dat[:, -1]
        file1 = open('../datasets/car/data_names.csv', 'r')
        self.features = [x.split(',')[0] for x in file1.read().split("\n")[:-1]]
        self.classes = np.unique(self.y)

    def hot_encoding(self):
        self.xenc = OneHotEncoder(handle_unknown='ignore')
        self.x_hot = self.xenc.fit_transform(self.X)
        self.yenc = OneHotEncoder(handle_unknown='ignore')
        self.yenc.fit(np.array(np.unique(self.y)).reshape(1, -1))

    def build_forest(self):
        f_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.forest = f_classifier.fit(self.x_hot, self.y)


if __name__ == '__main__':
    dataset = Dataset('car')
    rf, ency, classes = dataset.forest, dataset.yenc, dataset.classes
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
    rules = ruleMap.eigs()
    readableRule = RuleExtractor.RuleClass(dataset)
    readableRule.findRule(rules)

    print("end all")
