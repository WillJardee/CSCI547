from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree as Tree
import numpy as np


# rule extraction from tree via https://mljar.com/blog/extract-rules-decision-tree/

def tree_to_code2(tree):
    tree_ = tree.tree_
    print('start')

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
    print('done')
    return rec


# def tree_to_code(tree, feature_names):
#     tree_ = tree.tree_
#     feature_name = [
#         feature_names[i] if i != Tree._tree.TREE_UNDEFINED else "undefined!"
#         for i in tree_.feature
#     ]
#     feature_names = [f.replace(" ", "_")[:-5] for f in feature_names]
#     print("def predict({}):".format(", ".join(feature_names)))
#
#     def recurse(node, depth):
#         indent = "    " * depth
#         if tree_.feature[node] != Tree._tree.TREE_UNDEFINED:
#             name = feature_name[node]
#             threshold = tree_.threshold[node]
#             print("{}if {} <= {}:".format(indent, name, np.round(threshold, 2)))
#             recurse(tree_.children_left[node], depth + 1)
#             print("{}else:  # if {} > {}".format(indent, name, np.round(threshold, 2)))
#             recurse(tree_.children_right[node], depth + 1)
#         else:
#             print("{}return {}".format(indent, tree_.value[node]))
#
#     recurse(0, 1)
#
# def get_rules(tree, feature_names, class_names):
#     tree_ = tree.tree_
#     feature_name = [
#         feature_names[i] if i != Tree._tree.TREE_UNDEFINED else "undefined!"
#         for i in tree_.feature
#     ]
#
#     paths = []
#     path = []
#
#     def recurse(node, path, paths):
#
#         if tree_.feature[node] != Tree._tree.TREE_UNDEFINED:
#             name = feature_name[node]
#             threshold = tree_.threshold[node]
#             p1, p2 = list(path), list(path)
#             p1 += [f"({name} <= {np.round(threshold, 3)})"]
#             recurse(tree_.children_left[node], p1, paths)
#             p2 += [f"({name} > {np.round(threshold, 3)})"]
#             recurse(tree_.children_right[node], p2, paths)
#         else:
#             path += [(tree_.value[node], tree_.n_node_samples[node])]
#             paths += [path]
#
#     recurse(0, path, paths)
#
#     # sort by samples count
#     samples_count = [p[-1][1] for p in paths]
#     ii = list(np.argsort(samples_count))
#     paths = [paths[i] for i in reversed(ii)]
#
#     rules = []
#     for path in paths:
#         rule = "if "
#
#         for p in path[:-1]:
#             if rule != "if ":
#                 rule += " and "
#             rule += str(p)
#         rule += " then "
#         if class_names is None:
#             rule += "response: " + str(np.round(path[-1][0][0][0], 3))
#         else:
#             classes = path[-1][0][0]
#             l = np.argmax(classes)
#             rule += f"class: {class_names[l]} (proba: {np.round(100.0 * classes[l] / np.sum(classes), 2)}%)"
#         rule += f" | based on {path[-1][1]:,} samples"
#         rules += [rule]
#
#     return rules
