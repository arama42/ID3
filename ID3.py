from node import Node
from collections import deque


def replace_missing_values(examples):
    """Replaces missing values with the most common value of the concerned attribute."""
    most_common_value_by_attr = {}
    attr_val_frequencies = {}
    for example in examples:
        for attr, attr_val in example.items():
            if attr not in attr_val_frequencies:
                attr_val_frequencies[attr] = {}
            if attr_val != '?':
                if attr_val not in attr_val_frequencies[attr]:
                    attr_val_frequencies[attr][attr_val] = 0
                attr_val_frequencies[attr][attr_val] += 1
        for attr, freq_dict in attr_val_frequencies.items():
            max_freq = 0
            max_freq_attr_val = None
            for attr_val, freq in freq_dict.items():
                if freq > max_freq:
                    max_freq = freq
                    max_freq_attr_val = attr_val
            most_common_value_by_attr[attr] = max_freq_attr_val
    for example in examples:
        attrs_with_missing_values = []
        for attr, attr_val in example.items():
            if attr_val == '?':
                attrs_with_missing_values.append(attr)
        for attr in attrs_with_missing_values:
            example[attr] = most_common_value_by_attr[attr]


def ID3(examples, default):
    """
    Takes in an array of examples, and returns a tree (an instance of Node)
    trained on the examples.  Each example is a dictionary of attribute:value pairs,
    and the target class variable is a special attribute with the name "Class".
    Any missing attributes are denoted with a value of "?"
    """
    if len(examples) == 0:
        return None

    # Replace missing values with most common value for that attribute
    replace_missing_values(examples)

    # Create a node
    node = Node(examples)

    # Return a single node tree if it's a leaf or if there are no attributes left
    if node.is_leaf_node() or len(node.attributes) == 0:
        return node

    # Find out the best attribute to split on, based on maximum information gain
    max_information_gain_attr, max_information_gain_split = node.find_best_split()
    node.attr_to_split_on = max_information_gain_attr

    # Recursively build the tree (top-down)
    node.children = {}
    for attr_val, examples_with_attr_val in max_information_gain_split.items():
        node.children[attr_val] = ID3(examples_with_attr_val, default)

    return node


def get_bottom_up_tree(root):
    """Performs a level-order traversal of the tree and returns a reversed list of nodes."""
    bottom_up_tree = []
    q = deque()
    q.appendleft(root)
    while len(q) > 0:
        node = q.pop()
        bottom_up_tree.append(node)
        for _, child in node.children.items():
            q.appendleft(child)
    return list(reversed(bottom_up_tree))


def prune(root, examples):
    """
    Takes in a trained tree and a validation set of examples.  Prunes nodes in order
    to improve accuracy on the validation data; the precise pruning strategy is up to you.
    We used REDUCED ERROR PRUNING.
    """
    bottom_up_tree = get_bottom_up_tree(root)
    for node in bottom_up_tree:
        # Examine the nodes from the bottom-up
        children_attr_vals = list(node.children.keys())
        for attr_val in children_attr_vals:
            # Calculate validation accuracy before and after pruning a branch
            val_accuracy = test(root, examples)
            node.children[attr_val].is_pruned = True
            val_accuracy_pruned = test(root, examples)
            # Restore the branch if validation accuracy does not improve
            if val_accuracy_pruned < val_accuracy:
                node.children[attr_val].is_pruned = False
                

def test(node, examples):
    """
    Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
    of examples the tree classifies correctly).
    """
    num_correct = 0
    for example in examples:
        prediction = evaluate(node, example)
        if prediction == example['Class']:
            num_correct += 1
    return num_correct / len(examples)


def evaluate(node, example):
    """
    Takes in a tree and one example.  Returns the Class value that the tree
    assigns to the example.
    """
    if node.is_leaf_node() or len(node.attributes) == 0 or not node.has_traversable_child():
        return node.label
    attr_value = example[node.attr_to_split_on]
    if attr_value == '?':
        most_populous_branch = None
        most_populous_branch_size = 0
        for _, child in node.children.items():
            if child.is_pruned:
                continue
            branch_size = len(child.examples)
            if branch_size > most_populous_branch_size:
                most_populous_branch = child
                most_populous_branch_size = branch_size
        return evaluate(most_populous_branch, example)
    if attr_value not in node.children or node.children[attr_value].is_pruned:
        return node.label
    return evaluate(node.children[attr_value], example)
