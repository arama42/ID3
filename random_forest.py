import math
import statistics

import ID3
import parse
import random


def test_id3_single_tree(train_data, test_data):
    tree = ID3.ID3(train_data, '0')
    acc = ID3.test(tree, test_data)
    return acc


def test_random_forest(train_data, test_data, n_trees):
    attributes = set(train_data[0].keys())
    attributes.remove('Class')
    attributes = list(attributes)
    n_examples = len(train_data)

    trees = []
    for i_tree in range(n_trees):
        train_data_subsample = random.choices(train_data, k=n_examples)
        data_with_attr_removed = remove_attributes(attributes, train_data_subsample)
        tree = ID3.ID3(data_with_attr_removed, '0')
        trees.append(tree)

    num_correct = 0
    for example in test_data:
        rf_prediction = make_rf_prediction(example, trees)
        if rf_prediction == example['Class']:
            num_correct += 1
    return num_correct / len(test_data)


def remove_attributes(attributes, train_data_subsample):
    n_attr = len(attributes)
    n_attr_to_keep = math.ceil(math.sqrt(n_attr)) + 1
    attrs_to_remove = random.sample(attributes, k=n_attr-n_attr_to_keep)
    data_with_attr_removed = []
    for example in train_data_subsample:
        example_copy = example.copy()
        for attr in attrs_to_remove:
            example_copy.pop(attr)
        data_with_attr_removed.append(example_copy)
    return data_with_attr_removed


def make_rf_prediction(example, trees):
    predictions = {}
    for tree in trees:
        prediction = ID3.evaluate(tree, example)
        if prediction not in predictions:
            predictions[prediction] = 0
        predictions[prediction] += 1
    rf_prediction = None
    max_frequency = 0
    for prediction, frequency in predictions.items():
        if frequency > max_frequency:
            rf_prediction = prediction
            max_frequency = frequency
    return rf_prediction


def compare():
    data = parse.parse('candy.data')
    id3_accuracies = []
    rf_accuracies = []
    iterations = 500
    n_trees = 50
    for i in range(iterations):
        if (i + 1) % 50 == 0:
            print("Iteration {}/{}".format(i + 1, iterations))
        random.shuffle(data)
        train_data = data[:4 * len(data) // 5]
        test_data = data[4 * len(data) // 5:]
        id3_single_tree_acc = test_id3_single_tree(train_data, test_data)
        random_forest_acc = test_random_forest(train_data, test_data, n_trees)
        # print("ID3 single tree test accuracy:", id3_single_tree_acc)
        # print("Random forest test accuracy:", random_forest_acc)
        id3_accuracies.append(id3_single_tree_acc)
        rf_accuracies.append(random_forest_acc)

    print()
    print("ID3 single tree mean test accuracy:", statistics.mean(id3_accuracies))
    print("Random forest mean test accuracy:", statistics.mean(rf_accuracies))
    print()
    print("ID3 single tree test accuracy standard deviation:", statistics.stdev(id3_accuracies))
    print("Random forest test accuracy standard deviation:", statistics.stdev(rf_accuracies))


if __name__ == '__main__':
    compare()
