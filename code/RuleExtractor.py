import numpy as np


def normed(x): return x / np.linalg.norm(x)


class RuleClass:
    def __init__(self, dataset):
        self.dataset = dataset
        self.positive = []
        self.negative = []
        self.classes = []
        self.rule = []

    def findRule(self, rules):
        for ruleNumber in range(len(rules)):
            self.positive.append(
                self.dataset.xenc.inverse_transform(np.array(rules[ruleNumber][1][:21]).reshape(1, -1))[0])
            self.negative.append(
                self.dataset.xenc.inverse_transform(np.array(rules[ruleNumber][1][21:42]).reshape(1, -1))[0])
            self.classes.append(
                self.dataset.yenc.inverse_transform(np.array(rules[ruleNumber][1][-4:]).reshape(1, -1))[0])
            self.writeRule(ruleNumber)
        print(self.rule)

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


class LorentzMap:
    def __init__(self, num_pars, num_classes):
        self.n_pars = num_pars
        self.n_class = num_classes
        self.n_tot = 2 * num_pars + num_classes
        self.map = np.zeros((self.n_tot, self.n_tot))

    def add_term(self, x):
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

    def eigs(self, k=7):
        w, v = np.linalg.eig(self.map)

        def vecy(eig):
            xp, cp = np.abs(normed(eig[:-1 * self.n_class])), np.abs(normed(eig[-1 * self.n_class:]))
            return np.concatenate(([1 if i >= 1.5 * np.mean(xp) else 0 for i in xp],
                                   normed([i if i >= 1 / cp.size ** (1 / 2) else 0 for i in cp]) ** 2))

        w = normed(w ** 2)
        dic = {w[i]: vecy(v[i]) for i in range(w.size)}
        rules = [(x, dic[x]) for x in list(reversed(sorted(dic)))]

        picked_rules = []
        class_check = np.zeros(self.n_class)
        for i in range(k):
            picked_rules.append(rules[i])
            class_check += rules[i][1][-1 * self.n_class:]
        missing = []
        for i in range(self.n_class):
            if class_check[i] == 0:
                missing.append(i)
        while len(missing) != 0:
            i = missing[0]
            for j in rules[k + 1:]:
                if bool(j[1][-1 * (self.n_class - i)]):
                    picked_rules.append(j)
                    for a in range(self.n_class):
                        if bool(j[1][-1 * (self.n_class - a)]):
                            if a in missing:
                                missing.remove(a)
                    break

        return picked_rules
