import numpy as np


def normed(x): return x / (l) if (l := sum([y ** 2 for y in x]) ** (1 / 2)) else np.zeros(len(x))


class RuleClass:
    def __init__(self, dataset):
        self.dataset = dataset
        self.raw_rules_pos = []
        self.raw_rules_neg = []
        self.raw_class = []
        self.positive = []
        self.negative = []
        self.classes = []
        self.rule = []
        self.n_rules = None

    def findRule(self, rules):
        self.n_rules = len(rules)
        for ruleNumber in range(self.n_rules):
            self.positive.append(
                self.dataset.xenc.inverse_transform(
                    np.array(rules[ruleNumber][1][:self.dataset.encodeNumber]).reshape(1, -1))[0])
            self.negative.append(
                self.dataset.xenc.inverse_transform(
                    np.array(rules[ruleNumber][1][self.dataset.encodeNumber:self.dataset.encodeNumber * 2]).reshape(1,
                                                                                                                    -1))[
                    0])
            self.classes.append(
                self.dataset.yenc.inverse_transform(
                    np.array(rules[ruleNumber][1][-self.dataset.n_classes:]).reshape(1, -1))[0])
            self.writeRule(ruleNumber)

            self.raw_rules_pos.append(rules[ruleNumber][1][:self.dataset.encodeNumber])
            self.raw_rules_neg.append(rules[ruleNumber][1][self.dataset.encodeNumber:self.dataset.encodeNumber * 2])
            self.raw_class.append(rules[ruleNumber][1][-self.dataset.n_classes:])

    def writeRule(self, ruleNumber):
        temp = ""
        for i in range(len(self.positive[ruleNumber])):
            if self.positive[ruleNumber][i] is not None:
                if self.negative[ruleNumber][i] == self.positive[ruleNumber][i]:
                    continue
                if len(temp) != 0:
                    temp = temp + "and "
                temp = temp + "(" + str(self.dataset.features[i]) + "=" + str(self.positive[ruleNumber][i]) + ") "
            if self.negative[ruleNumber][i] is not None:
                if len(temp) != 0:
                    temp = temp + "and "
                temp = temp + "-" + "(" + str(self.dataset.features[i]) + "=" + str(self.negative[ruleNumber][i]) + ") "

        temp = temp + " --> "
        tempLength = len(temp)
        for j in self.classes[ruleNumber]:
            if j is not None:
                if len(temp) != tempLength:
                    temp = temp + "or "
                temp = temp + str(j) + " "
        self.rule.append(temp)

    def rule_check(self, x, y, weight_fun=lambda x: x):
        """

        :param (list) x: Input vector (NOT one hot-encoded)
        :param (str) y: Class
        :param (function) weight_fun:
        :return:
        """
        x_hot = self.dataset.xenc.transform(np.array(x).reshape([1, -1])).toarray()[0]
        lambs, match = [], []
        for i in range(self.n_rules):
            pos_dot = sum(normed(x_hot) * normed(self.raw_rules_pos[i]))
            neg_dot = sum(
                (np.ones(self.dataset.encodeNumber) - x_hot) / (sum([x ** 2 for x in x_hot]) ** (1 / 2)) * normed(
                    self.raw_rules_neg[i])) / (len(x) - 1)
            lambs.append(pos_dot + neg_dot)
            match.append(1 if y in self.classes[i] else -1)
        measure = []
        tot_weight = []
        for i in range(self.n_rules):
            measure.append(weight_fun(lambs[i]) * (match[i]))
            tot_weight.append(weight_fun(lambs[i]))
        meas = sum(measure) / sum(tot_weight)
        return meas


class LorentzMap:
    def __init__(self, num_pars, num_classes):
        self.n_pars = num_pars
        self.n_class = num_classes
        self.n_tot = 2 * num_pars + num_classes
        self.map = np.zeros((self.n_tot, self.n_tot))

    def add_term(self, x):
        """
        Adds a term to teh covariance matrix of the forest.

        :param x: rule vector to be added to the covariance matrix
        :return: None
        """

        counts = []
        for i in range(len(x) - self.n_class):
            if x[i] == -1:
                continue
            counts.append(i if x[i] == 1 else i + self.n_pars)
        for i in range(self.n_class):
            if x[i + self.n_pars] == -1:
                continue
            counts.append(i + 2 * self.n_pars)
        counts = [(x, y) for x in counts for y in counts]
        for i in counts: self.map[i] += 1

    def gen_rules(self, k=None, kstar=None,
                  x_fun=lambda i, x: i >= sorted(x)[int(len(x) * .75)],
                  y_fun=lambda i, x: i >= 1 / x.size ** (1 / 2)):
        """
        Generates the k* most important rules of the dataset

        :param (int) k: number of rules to extract. Default: number of features
        :param (function) x_fun: cut criteria for normalized input. Default: 1st quartile
        :param (function) y_fun: cut criteria for normalized class. Default: 1/sqrt(num of classes) - better than random
        :return: list of first k rules and any additional rules needed to fully cover the space.
        """

        if k is None:
            k = 15
        if kstar is None:
            kstar = 1

        w, v = np.linalg.eig(self.map)

        def vecy(eig):
            xp, cp = np.abs(normed(eig[:-1 * self.n_class])), np.abs(normed(eig[-1 * self.n_class:]))
            return np.concatenate(([1 if x_fun(i, xp) else 0 for i in xp],
                                   normed([i if y_fun(i, cp) else 0 for i in cp]) ** 2))

        w = normed(w ** 2)
        dic = {w[i]: vecy(v[i]) for i in range(w.size)}
        rules = [(x, dic[x]) for x in list(reversed(sorted(dic)))]

        picked_rules = []
        class_check = np.zeros(self.n_class)
        for i in range(k):
            picked_rules.append(rules[i])
            class_check += [1 if x > 0 else 0 for x in rules[i][1][-1 * self.n_class:]]
        missing = []
        for i in range(self.n_class):
            if class_check[i] < kstar:
                missing.append([i] * int(kstar - class_check[i]))
        while len(missing) != 0:
            i = missing[0][0]
            for j in rules[k + 1:]:
                if bool(j[1][-1 * (self.n_class - i)]):
                    picked_rules.append(j)
                    for a in range(self.n_class):
                        if bool(j[1][-1 * (self.n_class - a)]):
                            for b in missing:
                                if b[0] == a:
                                    b.pop(0)
                                    if not b:
                                        missing.remove([])
                    break

        return picked_rules
