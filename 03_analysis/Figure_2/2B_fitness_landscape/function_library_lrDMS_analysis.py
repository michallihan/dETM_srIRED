"""
Library of functions that are required for the lrDMS ETM analysis scripts. Keep in same folder.
"""

import numpy as np
import networkx as nx
import re


def count_hyphens(value):
    if value != "WT":
        return value.count("-") + 1
    else:
        return 0


def define_mutation_ids(dataframe):
    reference_sequence = dataframe.loc[dataframe["WT"] == True, "aa_seq"].tolist()

    if len(reference_sequence) != 1:
        raise ValueError("There should be exactly one reference sequence.")

    reference = reference_sequence[0]
    variants = dataframe.loc[dataframe["WT"] != True, "aa_seq"].tolist()

    def define_mutation_indices(reference_seq, variants_list):
        mut_arr = np.array([[ord(char) for char in string] for string in variants_list]).T
        ref_arr = np.array([ord(char) for char in reference_seq]).reshape(-1, 1)
        diff = mut_arr - ref_arr

        mut_idx = [np.where(diff[:, col] != 0)[0].tolist() for col in range(diff.shape[1])]

        return mut_idx

    def define_mutation_name(sequence, mut_idx, ref):
        ID = ""
        c = 0
        for index in mut_idx:
            old_aa = ref[index]
            loc = str(index + 1)
            new_aa = sequence[index]
            if c == 0:
                ID = ID + old_aa + loc + new_aa
            if c > 0:
                ID = ID + "-" + old_aa + loc + new_aa
            c = c + 1
        return ID

    mutation_indices = define_mutation_indices(reference, variants)
    my_list = ["WT"]

    for i in range(0, len(variants)):
        seq = variants[i]
        mut_index = mutation_indices[i]
        mut_name = define_mutation_name(seq, mut_index, reference)
        my_list.append(mut_name)

    return my_list

def extract_numbers(text):
    # Use the regex pattern \d+ to find all sequences of digits in the string
    numbers = re.findall(r"\d+", text)
    # Convert the extracted sequences from strings to integers
    return [int(num) for num in numbers]

def mutation_subract(list1, list2):
    return "".join(set(list1) ^ set(list2))

def list_subtract(list1,list2):
    return list(set(list1)-set(list2))

def generate_linear_sequence(a, b, start, stop, num):
    x_values = np.linspace(start, stop, num)
    y_values = a * x_values + b

    return x_values, y_values


def is_one_mutation_apart(seq1, seq2):
    if len(seq1) != len(seq2):
        return False
    diff_count = sum(1 for a, b in zip(seq1, seq2) if a != b)

    return diff_count == 1


def remove_unreachable_nodes(networkx_graph, target_node):
    # Perform a breadth-first search (BFS) from the target node to find all reachable nodes
    reachable_nodes = set(nx.single_source_shortest_path_length(networkx_graph, target_node).keys())
    all_nodes = set(networkx_graph.nodes())
    unreachable_nodes = all_nodes - reachable_nodes
    networkx_graph.remove_nodes_from(unreachable_nodes)

    return networkx_graph
