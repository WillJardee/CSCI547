from sklearn import tree as Tree


# rule extraction from tree via https://mljar.com/blog/extract-rules-decision-tree/
def tree_to_code2(tree):
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
