import math


class Node:
    def __init__(self, examples):
        self.examples = examples
        self.label = self.find_majority_label()
        self.attributes = self.get_attributes_from_examples()
        self.attr_to_split_on = None
        self.children = {}
        self.information = self.compute_information()
        self.is_pruned = False

    def is_leaf_node(self):
        classes_present = set()
        for example in self.examples:
            classes_present.add(example['Class'])
        return len(classes_present) == 1

    def has_traversable_child(self):
        for _, child in self.children.items():
            if not child.is_pruned:
                return True
        return False

    def find_majority_label(self):
        class_frequencies = {}
        max_frequency = 0
        max_frequency_cls = self.examples[0]['Class']
        for example in self.examples:
            cls = example['Class']
            if cls not in class_frequencies:
                class_frequencies[cls] = 0
            class_frequencies[cls] += 1
            if class_frequencies[cls] > max_frequency:
                max_frequency = class_frequencies[cls]
                max_frequency_cls = cls
        return max_frequency_cls

    def get_attributes_from_examples(self):
        attributes = set()
        for example in self.examples:
            for attr in example.keys():
                if attr != 'Class':
                    attributes.add(attr)
        return attributes

    def compute_information(self):
        class_frequencies = {}
        for example in self.examples:
            cls = example['Class']
            if cls not in class_frequencies:
                class_frequencies[cls] = 0
            class_frequencies[cls] += 1
        information = 0
        for _, frequency in class_frequencies.items():
            p = frequency / len(self.examples)
            information += -(p * math.log2(p))
        return information

    def find_best_split(self):
        max_information_gain = 0
        max_information_gain_attr = None
        max_information_gain_split = None
        for attr in self.attributes:
            examples_by_attr_val = {}
            for example in self.examples:
                if example[attr] not in examples_by_attr_val:
                    examples_by_attr_val[example[attr]] = []
                example_with_attr_removed = example.copy()
                example_with_attr_removed.pop(attr)
                examples_by_attr_val[example[attr]].append(example_with_attr_removed)
            information_gain = self.information
            for attr_val, examples_with_attr_val in examples_by_attr_val.items():
                temp_node = Node(examples_with_attr_val)
                num_of_examples = len(examples_with_attr_val)
                weight = num_of_examples / len(self.examples)
                information_gain -= weight * temp_node.information
            if max_information_gain_attr is None or information_gain > max_information_gain or (
                    max_information_gain == 0 and is_split_non_trivial(examples_by_attr_val)):
                max_information_gain = information_gain
                max_information_gain_attr = attr
                max_information_gain_split = examples_by_attr_val
        return max_information_gain_attr, max_information_gain_split


def is_split_non_trivial(examples_by_attr_val):
    num_of_branches = len(examples_by_attr_val)
    num_of_empty_branches = 0
    for attr_val, examples in examples_by_attr_val.items():
        if len(examples) == 0:
            num_of_empty_branches += 1
    return num_of_empty_branches < num_of_branches - 1
