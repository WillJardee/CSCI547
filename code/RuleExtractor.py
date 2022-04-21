import pandas as pd
import numpy as np


class LorentzMap:
    def __int__(self, num_pars, num_classes):
        self.map = np.zeros((num_pars*2, num_pars*2))
        self.n_pars = num_pars
        self.n_class = num_classes
        self.n_tot = num_pars + num_classes

    def add_term(self, x):
        counts = []
        for i in range(len(x)):
            if x[i] == -1: continue
            counts.append(x[i] if x[i] == 1 else x[i]+self.n_tot)
        counts = [(x, y) for x in counts for y in counts]
        for i in counts: self.map[i] += 1
