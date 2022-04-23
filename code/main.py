from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree as Tree
import RuleExtractor
import numpy as np
import TreeRuleExtractor as tRule
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, data_name):
        self.n_feats, self.n_classes = None, None
        self.X_train, self.y_train, self.X_test, self.y_test, self.y, self.X, self.features, self.classes = \
            None, None, None, None, None, None, None, None
        self.xenc, self.x_hot, self.yenc = None, None, None
        self.forest = None
        self.encodeNumber = 0
        self.get_dat(data_name)
        self.hot_encoding()
        self.build_forest()

    def get_dat(self, data_name):
        dat = np.genfromtxt("../datasets/" + data_name + "/data.csv", delimiter=",", dtype=str)
        self.X, self.y = dat[:, 0:-1], dat[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size= 0.3, shuffle=True)
        file1 = open('../datasets/' + data_name + '/data_names.csv', 'r')
        self.features = [x.split(',')[0] for x in file1.read().split("\n")[:-1]]
        self.n_feats = len(self.features)
        self.classes = np.unique(self.y)
        self.n_classes = len(self.classes)

    def hot_encoding(self):
        self.xenc = OneHotEncoder(handle_unknown='ignore')
        self.x_hot = self.xenc.fit(self.X).transform(self.X_train)
        self.yenc = OneHotEncoder(handle_unknown='ignore')
        for categories in self.xenc.categories_:
            self.encodeNumber += len(categories)
        self.yenc.fit(np.array(np.unique(self.y_train)).reshape(1, -1))

    def build_forest(self):
        f_classifier = RandomForestClassifier(n_estimators=1000, random_state=42)
        self.forest = f_classifier.fit(self.x_hot, self.y_train)


if __name__ == '__main__':
    for numberofTest in range(15):
        dataset = Dataset('tic-tac-toe')
        rf, ency, classes = dataset.forest, dataset.yenc, dataset.classes
        vector = []
        for each_tree in range(len(rf)):
            rule = tRule.tree_to_code2(rf[each_tree])
            for each_rule in range(len(rule)):
                arr = -1*np.ones(dataset.encodeNumber)
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

        ruleMap = RuleExtractor.LorentzMap(dataset.encodeNumber, classes.shape[0])

        for each_vector in range(len(vector)):
            ruleMap.add_term(vector[each_vector])

        print("end map gen")
        rules = ruleMap.gen_rules()
        readableRule = RuleExtractor.RuleClass(dataset)
        readableRule.findRule(rules)

        for i in readableRule.rule: print(i)

        trainResult = []
        for index in range(len(dataset.X_train)):
            trainResult.append(readableRule.rule_check(dataset.X_train[index], dataset.y_train[index]))
        print(trainResult)

        testResult = []
        for index in range(len(dataset.X_test)):
            testResult.append(readableRule.rule_check(dataset.X_test[index], dataset.y_test[index]))
        print(testResult)

        plt.boxplot([trainResult, testResult], labels=['train', 'test'], vert=False)
        # plt.boxplot(testResult, labels=['test'])
        plt.legend()
        plt.show()

        print("end all")
