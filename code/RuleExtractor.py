import numpy as np


class LorentzMap:
    def __init__(self, num_pars, num_classes):
        self.n_pars = num_pars
        self.n_class = num_classes
        self.n_tot = 2 * num_pars + num_classes
        self.map = np.zeros((self.n_tot, self.n_tot))

    def add_term(self, x):
        counts = []
        for i in range(len(x) - self.n_class):
            if x[i] == -1: continue
            counts.append(i if x[i] == 1 else i + self.n_pars)
        for i in range(self.n_class):
            if x[i + self.n_pars] == -1: continue
            counts.append(i + 2 * self.n_pars)
        counts = [(x, y) for x in counts for y in counts]
        for i in counts: self.map[i] += 1

    def eigs(self):
        w, v = np.linalg.eig(self.map)

        def vecy(eig):
            normed = lambda x: x / np.linalg.norm(x)
            xp, cp = np.abs(normed(eig[:-4])), np.abs(normed(eig[-4:]))
            return np.concatenate(([1 if i >= 1 / xp.size ** (1 / 2) else 0 for i in xp],
                                   [1 if i >= 1 / cp.size ** (1 / 2) else 0 for i in cp]))

        dic = {w[i] ** (1 / 2): vecy(v[i]) for i in range(w.size)}
        return [dic[x] for x in list(reversed(sorted(dic)))]
