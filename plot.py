import statistics

import matplotlib.pyplot as plt
import seaborn as sns

import ID3
import parse
import random


def plot_house_file_decision_tree():
    max_train_set_size = 300
    without_pruning_by_train_size = {}
    with_pruning_by_train_size = {}
    data = parse.parse('house_votes_84.data')
    random.shuffle(data)
    test = data[max_train_set_size:]
    data_minus_test = data[:max_train_set_size]
    for train_set_size in range(10, max_train_set_size + 1, 2):
        valid_set_size = train_set_size * 1 // 5
        without_pruning = []
        with_pruning = []
        print('Train set size: {}'.format(train_set_size))
        for i in range(100):
            random.shuffle(data_minus_test)
            train = data_minus_test[:train_set_size - valid_set_size]
            valid = data_minus_test[train_set_size - valid_set_size:train_set_size]

            tree = ID3.ID3(train, 'democrat')
            acc = ID3.test(tree, test)
            without_pruning.append(acc)

            ID3.prune(tree, valid)
            acc = ID3.test(tree, test)
            with_pruning.append(acc)

        without_pruning_by_train_size[train_set_size] = statistics.mean(without_pruning)
        with_pruning_by_train_size[train_set_size] = statistics.mean(with_pruning)
    sns.set()
    sns.lineplot({'Without Pruning': without_pruning_by_train_size,
                  'With Pruning': with_pruning_by_train_size})
    plt.xlabel('train_set_size')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == '__main__':
    plot_house_file_decision_tree()
