from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree as Tree
import RuleExtractor
import numpy as np
import TreeRuleExtractor as tRule
import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, data_name, f_size_val=100):
        self.n_feats, self.n_classes = None, None
        self.X_train, self.y_train, self.X_test, self.y_test, self.y, self.X, self.features, self.classes = \
            None, None, None, None, None, None, None, None
        self.xenc, self.x_hot, self.yenc = None, None, None
        self.forest = None
        self.encodeNumber = 0

        self.get_dat(data_name)
        self.hot_encoding()
        self.build_forest(f_size_val)

    def get_dat(self, data_name):
        dat = np.genfromtxt("../datasets/" + data_name + "/data.csv", delimiter=",", dtype=str)
        self.X, self.y = dat[:, 0:-1], dat[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
                                                                                shuffle=True)
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

    def build_forest(self, f_size_val=100):
        f_classifier = RandomForestClassifier(n_estimators=f_size_val, random_state=42)
        self.forest = f_classifier.fit(self.x_hot, self.y_train)


def run_test(dat_s, k_val, kstar_val, f_size_val, runs=15, save=True, plot=False):
    for numberTest in range(runs):
        dataset = Dataset(dat_s, f_size_val)
        rf, ency, classes = dataset.forest, dataset.yenc, dataset.classes
        vector = []
        for each_tree in range(len(rf)):
            rule = tRule.tree_to_code2(rf[each_tree])
            for each_rule in range(len(rule)):
                arr = -1 * np.ones(dataset.encodeNumber)
                for j in rule[each_rule][:-1]:
                    arr[j[0] - 1] = j[1]
                temp = rule[each_rule][-1]
                for j in range(len(temp[0])):
                    if temp[0][j] != 0:
                        temp[0][j] = 1
                    else:
                        temp[0][j] = -1
                arr = np.append(arr, [temp])
                vector.append(arr)

        rule_map = RuleExtractor.LorentzMap(dataset.encodeNumber, classes.shape[0])

        for each_vector in range(len(vector)):
            rule_map.add_term(vector[each_vector])

        rules = rule_map.gen_rules(k=k_val, kstar=kstar_val)
        readable_rule = RuleExtractor.RuleClass(dataset)
        readable_rule.findRule(rules)

        train_result_lin = []
        for index in range(len(dataset.X_train)):
            train_result_lin.append(readable_rule.rule_check(dataset.X_train[index], dataset.y_train[index]))

        test_result_lin = []
        for index in range(len(dataset.X_test)):
            test_result_lin.append(readable_rule.rule_check(dataset.X_test[index], dataset.y_test[index]))

        train_result_exp = []
        for index in range(len(dataset.X_train)):
            train_result_exp.append(readable_rule.rule_check(dataset.X_train[index], dataset.y_train[index],
                                                             weight_fun=lambda x: np.exp(x)))

        test_result_exp = []
        for index in range(len(dataset.X_test)):
            test_result_exp.append(readable_rule.rule_check(dataset.X_test[index], dataset.y_test[index],
                                                            weight_fun=lambda x: np.exp(x)))

        if plot:
            plt.boxplot([train_result_lin, test_result_lin], labels=['train', 'test'], vert=False)
            # plt.boxplot(testResult, labels=['test'])
            plt.legend()
            plt.show()

        if save:
            fil_name = str(k_val).zfill(2) + "_" + str(kstar_val) + "_" + str(f_size_val).zfill(4) + "_" + str(
                numberTest).zfill(2)
            fil_name = '../datasets/' + dat_s + "/runs/" + fil_name + "_"
            fil_hum = open(fil_name + "humanReadable.txt", 'w')
            fil_raw = open(fil_name + "raw.txt", 'w')
            fil_train_met = open(fil_name + "trainMetric.txt", 'w')
            fil_test_met = open(fil_name + "testMetric.txt", 'w')

            pream = str(k_val) + "," + str(kstar_val) + "," + str(f_size_val) + "\n"
            fil_hum.write(pream)
            fil_raw.write(pream)
            fil_train_met.write(pream)
            fil_test_met.write(pream)

            for j in readable_rule.rule: fil_hum.write(j + "\n")
            for j in range(len(readable_rule.rule)): fil_raw.write(str(readable_rule.raw_rules_pos[j]) +
                                                                   str(readable_rule.raw_rules_neg[j]) +

                                                                   str(readable_rule.raw_class[j]) + "\n")
            fil_train_met.write(str(dataset.forest.score(dataset.xenc.transform(dataset.X_train), dataset.y_train))
                                + "\n")
            fil_train_met.write("linear:\n")
            fil_train_met.write(str(train_result_lin) + "\n")
            fil_train_met.write("\nexponential:\n")
            fil_train_met.write(str(train_result_exp) + "\n")

            fil_test_met.write(str(dataset.forest.score(dataset.xenc.transform(dataset.X_test), dataset.y_test))
                                + "\n")
            fil_test_met.write("linear:\n")
            fil_test_met.write(str(test_result_lin) + "\n")
            fil_test_met.write("\nexponential:\n")
            fil_test_met.write(str(test_result_exp) + "\n")

            fil_hum.close()
            fil_raw.close()
            fil_train_met.close()
            fil_test_met.close()
        print(f'Done: k={k_val}, k*={kstar_val}, forest size={f_size_val}; number={numberTest + 1}/{runs}')
    print("-" * 50)


if __name__ == '__main__':
    dat_set = 'car'
    run = 5

    k = [0, 2, 6, 10, 20]
    kstar = [0, 1, 2, 3]
    f_size = [20, 50, 100, 200, 500, 1000, 2000, 5000]

    tests = [(x, y, z) for x in k for y in kstar for z in f_size]
    for i in f_size:
        tests.remove((0, 0, i))

    print("'''\n" +
          "Running on data set: " + dat_set + "\n" +
          f"k values:  {str(k)}\n" +
          f"k* values: {str(kstar)}\n" +
          f"forest sizes: {str(f_size)}\n" +
          f"{run} tests each\n" +
          "'''\n\n" +
          "Begin tests\n" +
          "-" * 30 + "\n")

    for i in tests:
        run_test(dat_set, *i, runs=run)
