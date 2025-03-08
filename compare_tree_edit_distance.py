import re
import numpy as np
import argparse
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def build_tree(sentence):
    tree = defaultdict(list)
    labels = {}
    root = None

    for node in sentence:
        labels[node.id] = (node.form, node.deprel)
        if node.head == 0:
            root = node.id
        else:
            tree[node.head].append(node.id)

    return tree, labels, root


def subtree_edit_distance(node1, node2, tree1, tree2, labels1, labels2, memo):
    if (node1, node2) in memo:
        return memo[(node1, node2)]

    cost = 0 if labels1[node1][1] == labels2[node2][1] else 1

    children1, children2 = tree1[node1], tree2[node2]
    len1, len2 = len(children1), len(children2)

    if len1 == 0 and len2 == 0:
        memo[(node1, node2)] = cost
        return cost

    dist_matrix = np.zeros((len1 + 1, len2 + 1))

    for i in range(len1 + 1):
        dist_matrix[i][0] = i
    for j in range(len2 + 1):
        dist_matrix[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            sub_cost = subtree_edit_distance(children1[i-1], children2[j-1], tree1, tree2, labels1, labels2, memo)
            dist_matrix[i][j] = min(
                dist_matrix[i-1][j] + 1,  # deletion
                dist_matrix[i][j-1] + 1,  # insertion
                dist_matrix[i-1][j-1] + sub_cost  # substitution
            )

    total_cost = cost + dist_matrix[len1][len2]
    memo[(node1, node2)] = total_cost
    return total_cost


def improved_tree_edit_distance(tree_a, tree_b):
    tree1, labels1, root1 = build_tree(tree_a)
    tree2, labels2, root2 = build_tree(tree_b)

    memo = {}
    ted = subtree_edit_distance(root1, root2, tree1, tree2, labels1, labels2, memo)
    normalized_ted = ted / max(len(tree_a), len(tree_b))

    return normalized_ted


class DependencyNode:
    def __init__(self, id: int, form: str, head: int, deprel: str = None):
        self.id = id
        self.form = form
        self.head = head
        self.deprel = deprel
        self.children = []

# Your original parse functions remain unchanged...
def parse_custom_format(text: str):
    sentences = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        current_sentence = []
        tokens = line.split()
        for i, token in enumerate(tokens):
            if '_' not in token or '-' not in token:
                continue
            word_deprel, head_str = token.rsplit('-', 1)
            word, deprel = word_deprel.rsplit('_', 1)
            deprel = deprel[:3]
            head = int(head_str) if head_str.isdigit() else 0
            current_sentence.append(DependencyNode(i+1, word, head, deprel))
        # Check for root presence
        if current_sentence and any(node.head == 0 for node in current_sentence):
            sentences.append(current_sentence)
    return sentences

def validate_tree(sentence):
    heads = {node.id: node.head for node in sentence}
    root_count = sum(1 for head in heads.values() if head == 0)
    if root_count != 1:
        return False
    visited, stack = set(), [next(node.id for node in sentence if node.head == 0)]
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        stack.extend(child.id for child in sentence if child.head == node_id)
    return len(visited) == len(sentence)


def parse_ud_format(text: str):
    sentences, current_sentence = [], []
    for line in text.strip().split('\n'):
        line = line.strip()
        if line.startswith('#') or not line:
            if line.startswith('# sent_id') and current_sentence:
                # Check for root presence
                if any(node.head == 0 for node in current_sentence):
                    sentences.append(current_sentence)
                current_sentence = []
            continue
        fields = line.split('\t')
        if len(fields) < 8:
            continue
        try:
            node_id, form, head, deprel = int(fields[0]), fields[1], int(fields[6]), fields[7][:3]
            current_sentence.append(DependencyNode(node_id, form, head, deprel))
        except ValueError:
            continue
    # Final check for the last sentence
    if current_sentence and any(node.head == 0 for node in current_sentence):
        sentences.append(current_sentence)
    return sentences

def is_flat_tree(sentence):
    """The parser will default to flat trees when it cannot meaningfully parse a sentence, so we use this filter to make sure that our reference treebanks do not contain flat trees. We apply this indiscriminately to all treebanks, therefore ensuring a fair comparison."""
    if not sentence:
        return False
    
    root_id = sentence[0].id
    is_flat = all(node.head == root_id or node.id == root_id for node in sentence)
    
    # Check for 'flat' or 'con' labels
    flat_con_count = sum(node.deprel in ('fla', 'con') for node in sentence)
    flat_con_ratio = flat_con_count / len(sentence)
    
    return is_flat or flat_con_ratio > 0.2

# Your existing tree_edit_distance_with_labels and compute_closest_distances remain unchanged...
def tree_edit_distance_with_labels(tree1, tree2):
    head_map1 = {node.id: (node.head, node.deprel) for node in tree1}
    head_map2 = {node.id: (node.head, node.deprel) for node in tree2}
    nodes1, nodes2 = set(head_map1), set(head_map2)
    common_nodes = nodes1.intersection(nodes2)
    ins_del_cost = len(nodes1.symmetric_difference(nodes2))
    rel_cost = sum((head_map1[n] != head_map2[n]) for n in common_nodes)
    return (ins_del_cost + rel_cost) / max(len(nodes1), len(nodes2))

def compute_closest_distances(input_sentences, treebanks):
    results = {}
    closest_sentences = {}
    for i, sentence in tqdm(enumerate(input_sentences), total=len(input_sentences), desc="Processing Sentences"):
        sentence_length = len(sentence)
        min_l, max_l = int(sentence_length * 0.1), int(sentence_length * 5)
        closest_distances = [float('inf')] * len(treebanks)
        closest_matches = [None] * len(treebanks)

        for idx, tb in enumerate(treebanks):
            distances, closest_sentence = compute_distances_for_treebank(sentence, tb, min_l, max_l)
            if distances:
                closest_distances[idx] = np.mean(distances[:1000])
                closest_matches[idx] = closest_sentence

        results[i] = tuple(closest_distances)
        closest_sentences[i] = closest_matches
    return results, closest_sentences

def compute_distances_for_treebank(sentence, treebank, min_l, max_l):
    filtered_tb = [s for s in treebank if min_l <= len(s) <= max_l] or treebank
    distances = sorted(
        ((tree_edit_distance_with_labels(sentence, ref_sentence), ref_sentence) 
        for ref_sentence in filtered_tb),  # Use tree_edit_distance_with_labels
        key=lambda x: x[0]  # Sort by the distance value
    )
    closest_sentence = distances[0][1] if distances else None
    return [d[0] for d in distances], closest_sentence

def print_tree(sentence):
    """Helper function to print the tree structure of a sentence."""
    def print_subtree(node_id, indent=0):
        node = next(node for node in sentence if node.id == node_id)
        print(' ' * indent + f"{node.form} ({node.deprel})")
        for child in sorted((n for n in sentence if n.head == node_id), key=lambda x: x.id):
            print_subtree(child.id, indent + 4)

    root = next(node.id for node in sentence if node.head == 0)
    print_subtree(root)

# Main script logic
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='tree_distances.txt')
    args = parser.parse_args()

    input_file = 'data/yd-parsed.txt'
    treebank_files = [
        'treebanks/la_proiel-ud-train.conllu',
        'treebanks/sa_vedic-ud-train.conllu',
        'treebanks/de_gsd-ud-train.conllu',
        'treebanks/lzh_kyoto-ud-train.conllu',
        'treebanks/is_icepahc-ud-train.conllu',
        'treebanks/sa_ufal-ud-test.conllu'
    ]

    with open(input_file, 'r', encoding='utf-8') as f:
        input_sentences = parse_custom_format(f.read())
    input_sentences = [s for s in input_sentences if validate_tree(s)]

    treebanks = []
    for file in treebank_files:
        with open(file, 'r', encoding='utf-8') as f:
            tb_sentences = [s for s in parse_ud_format(f.read()) if len(s) >= 3]
            tb_sentences = [s for s in tb_sentences if not is_flat_tree(s)]
            tb_sentences = [s for s in tb_sentences if validate_tree(s)]
            treebanks.append(tb_sentences)

    results, closest_sentences = compute_closest_distances(input_sentences, treebanks)

    # Calculate total distances for each treebank
    total_distances = np.sum([list(dists) for dists in results.values()], axis=0)

    # Calculate the mean distance
    mean_distance = np.mean(total_distances)

    # Calculate differences from the mean
    differences_from_mean = total_distances - mean_distance

    # Print differences from the mean to the console
    print("\nDifferences from Mean Distance:")
    for label, difference in zip(treebank_files, differences_from_mean):
        print(f"{label}: {difference:.4f}")

    # Use filenames as labels
    labels = [
        'la_proiel-ud-train.conllu',
        'sa_vedic-ud-train.conllu',
        'de_gsd-ud-train.conllu',
        'lzh_kyoto-ud-train.conllu',
        'is_icepahc-ud-train.conllu',
        'sa_ufal-ud-test.conllu'
    ]

    plt.figure(figsize=(12, 8))  # Increase figure size
    bars = plt.bar(labels, differences_from_mean, color='skyblue')  # Use a distinct color

    # Add data labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', va='bottom', ha='center')

    plt.ylabel('Difference from Mean Distance')
    plt.title('Difference from Mean Tree Edit Distance Across Corpora')
    plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.savefig('difference_from_mean_distances.png')

    with open(args.output, 'w', encoding='utf-8') as f:
        header = "Sentence_ID\t" + "\t".join(f"Closest_TB{i+1}" for i in range(6)) + "\n"
        f.write(header)
        for sid, dists in sorted(results.items()):
            f.write(f"{sid+1}\t" + "\t".join(f"{dist:.4f}" for dist in dists) + '\n')

    # Print total statistics to the console
    print("\nTotal Statistics:")
    for i, average_distance in enumerate(total_distances):
        print(f"Treebank {i+1}: {average_distance:.4f}")

    # Print all sentences that match with corpus 2
    do_print = True
    if do_print:
        print("\nSentences matching with Corpus 2:")
        for i, match in closest_sentences.items():
            if match[1]:  # Check if there is a match with corpus 2
                print(f"\nInput Sentence {i+1}:")
                print_tree(input_sentences[i])
                print("\nMatched Sentence from Corpus 2:")
                print_tree(match[1])

if __name__ == "__main__":
    main()
