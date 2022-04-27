import numpy as np

from sklearn import tree as Tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def get_a_tree(tree):
    tree_ = tree.tree_

    def recurse(node):
        if tree_.feature[node] != Tree._tree.TREE_UNDEFINED:
            for i in (lefts := recurse(tree_.children_left[node])):
                i.insert(0, (tree_.feature[node], 0))
            for i in (rights := recurse(tree_.children_right[node])):
                i.insert(0, (tree_.feature[node], 1))
            return lefts + rights
        else:
            return [[tree_.value[node]]]

    rec = recurse(0)
    return rec


class Dataset:
    def __init__(self, data_name, f_size_val=100):
        self.n_feats, self.n_classes = None, None
        self.X_train, self.y_train, self.X_test, self.y_test, self.y, self.X, self.features, self.classes = \
            None, None, None, None, None, None, None, None
        self.forest = None
        self.encodeNumber = 0
        self.rules = []

        self.get_dat(data_name)
        self.build_forest(f_size_val)
        self.get_all_rules()

    def get_dat(self, data_name):
        dat = np.genfromtxt("datasets/" + data_name, delimiter=",", dtype=int)
        self.X, self.y = dat[:, 0:-1], dat[:, -1]
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3,
        #                                                                         shuffle=True)

        self.X_train, self.y_train = self.X, self.y
        # file1 = open('../datasets/' + data_name + '/data_names.csv', 'r')
        # self.features = [x.split(',')[0] for x in file1.read().split("\n")[:-1]]
        self.n_feats = 4  # TODO: make arbitrary
        # self.classes =
        self.n_classes = 2

    def build_forest(self, f_size_val=100):
        f_classifier = RandomForestClassifier(n_estimators=f_size_val, random_state=42)
        self.forest = f_classifier.fit(self.X_train, self.y_train)

    def get_all_rules(self):
        for t in self.forest:
            self.rules.append(get_a_tree(t))


class LorentzMap:
    def __init__(self, num_pars, num_classes):
        self.n_pars = num_pars
        self.n_class = num_classes
        self.n_tot = num_pars + num_classes
        self.map = [np.zeros((self.n_pars, self.n_pars)) for i in range(self.n_class)]

    def add_forest(self, f):
        for tree in f:
            self.add_tree(tree)

    def add_tree(self, t):
        for term in t:
            for k in range(len(t[-1][0])):
                if t[-1][0][k] != 0:
                    self.add_term(term[:-1*self.n_class], k)

    def add_term(self, x, k):
        for i in x:
            for j in x:
                self.map[k][j[0], i[0]] += 1 if j[1] == 1 else -1


if __name__ == '__main__':
    dat = Dataset('adult+stretch_readable.data', f_size_val=100)
    map = LorentzMap(dat.n_feats, dat.n_classes)
    map.add_forest(dat.rules)

    # w, v = np.linalg.eig(map.map.T)
    # for i in range(len(w)):
    #     print(str(w[i]) + str(v[i]))

    pass
