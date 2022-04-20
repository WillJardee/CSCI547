from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def build_dataset(dataset):
    dat = np.genfromtxt("../datasets/" + dataset + "/data.csv", delimiter=",", dtype=str)
    X, y = dat[:, 0:-1], dat[:, -1]
    enc = OneHotEncoder(handle_unknown='ignore')

    rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    rf.fit(X, y)
    pass


if __name__ == '__main__':
    pass
