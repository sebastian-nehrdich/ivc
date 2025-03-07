import re
import numpy as np
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Any, Optional
import matplotlib.pyplot as plt
import random


class DependencyNode:
    """Represents a node in a dependency tree."""
    def __init__(self, id: int, form: str, head: int, deprel: str = None):
        self.id = id
        self.form = form
        self.head = head
        self.deprel = deprel
        self.children = []


class DependencyTree:
    """Represents a dependency tree."""
    def __init__(self, nodes: List[DependencyNode]):
        self.nodes = {node.id: node for node in nodes}
        self.root = None
        
        # Build the tree structure
        for node_id, node in self.nodes.items():
            if node.head == 0:
                self.root = node
            elif node.head in self.nodes:
                self.nodes[node.head].children.append(node)


def parse_custom_format(text: str) -> List[List[DependencyNode]]:
    """Parse the first format (non-UD) into a list of sentences."""
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
                
            # Extract the head information
            parts = token.split('-')
            if len(parts) < 2:
                continue
                
            word = parts[0]
            head_str = parts[1]
            
            try:
                head = int(head_str)
            except ValueError:
                # If head is not a valid integer, treat as root
                head = 0
                
            node = DependencyNode(i+1, word, head)
            current_sentence.append(node)
        
        if current_sentence:
            sentences.append(current_sentence)
        
    return sentences


def parse_ud_format(text: str) -> List[List[DependencyNode]]:
    """Parse the Universal Dependencies format into a list of sentences."""
    sentences = []
    current_sentence = []
    
    for line in text.strip().split('\n'):
        line = line.strip()
        if line.startswith('#') or not line:
            if line.startswith('# sent_id') and current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
            continue
        
        fields = line.split('\t')
        if len(fields) < 8:
            continue
            
        try:
            node_id = int(fields[0])
            form = fields[1]
            head = int(fields[6])
            deprel = fields[7]
            
            node = DependencyNode(node_id, form, head, deprel)
            current_sentence.append(node)
        except ValueError:
            # Skip lines with non-integer IDs or heads
            continue
            
    if current_sentence:
        sentences.append(current_sentence)
        
    return sentences


def tree_edit_distance(tree1: List[DependencyNode], tree2: List[DependencyNode]) -> float:
    """
    Calculate the tree edit distance between two dependency trees.
    
    This is a simplified version that calculates:
    1. Differences in parent-child relationships
    2. Differences in tree structure
    """
    # Create dictionaries mapping node IDs to their heads
    head_map1 = {node.id: node.head for node in tree1}
    head_map2 = {node.id: node.head for node in tree2}
    # Get sets of node IDs for both trees
    nodes1 = set(head_map1.keys())
    nodes2 = set(head_map2.keys())
    
    # Calculate the cost of node insertion/deletion
    insertion_deletion_cost = len(nodes1.symmetric_difference(nodes2))
    
    # Calculate the cost of changing parent-child relationships
    relationship_cost = 0
    common_nodes = nodes1.intersection(nodes2)
    
    for node_id in common_nodes:
        if head_map1[node_id] != head_map2[node_id]:
            relationship_cost += 1
    
    # The total edit distance is the sum of insertion/deletion and relationship costs
    total_distance = insertion_deletion_cost + relationship_cost
    
    # Normalize by the size of the larger tree to get a value between 0 and 1
    max_size = max(len(nodes1), len(nodes2))
    if max_size == 0:
        return 0.0
    
    # Explicit normalization
    normalized_distance = total_distance / max_size
    
    return normalized_distance


def safe_average(distances):
    return np.mean(distances) if distances else 1.0  # Using max normalized distance (1.0) as fallback

def compute_average_distances(input_sentences: List[List[DependencyNode]], 
                             treebank1: List[List[DependencyNode]], 
                             treebank2: List[List[DependencyNode]],
                             treebank3: List[List[DependencyNode]],
                             treebank4: List[List[DependencyNode]],
                             treebank5: List[List[DependencyNode]],
                             treebank6: List[List[DependencyNode]]) -> Dict[int, Tuple[float, float, float, float, float, float]]:
    """
    Compute the average tree edit distance between each input sentence and the six treebanks.
    
    Returns a dictionary mapping sentence indices to tuples of (avg_distance_to_treebank1, avg_distance_to_treebank2, avg_distance_to_treebank3, avg_distance_to_treebank4, avg_distance_to_treebank5, avg_distance_to_treebank6)
    """
    results = {}
    max_samples = 1000  # Maximum number of samples to use for averaging

    for i, sentence in enumerate(input_sentences):
        sentence_length = len(sentence)
        min_length = int(sentence_length * 0.8)
        max_length = int(sentence_length * 1.2)
        
        # Filter and sample treebank sentences by length
        filtered_treebank1 = [s for s in treebank1 if min_length <= len(s) <= max_length]
        filtered_treebank2 = [s for s in treebank2 if min_length <= len(s) <= max_length]
        filtered_treebank3 = [s for s in treebank3 if min_length <= len(s) <= max_length]
        filtered_treebank4 = [s for s in treebank4 if min_length <= len(s) <= max_length]
        filtered_treebank5 = [s for s in treebank5 if min_length <= len(s) <= max_length]
        filtered_treebank6 = [s for s in treebank6 if min_length <= len(s) <= max_length]
        # print length of filtered treebanks
        print(f"Length of filtered treebank1: {len(filtered_treebank1)}")
        print(f"Length of filtered treebank2: {len(filtered_treebank2)}")
        print(f"Length of filtered treebank3: {len(filtered_treebank3)}")
        print(f"Length of filtered treebank4: {len(filtered_treebank4)}")
        print(f"Length of filtered treebank5: {len(filtered_treebank5)}")
        print(f"Length of filtered treebank6: {len(filtered_treebank6)}")
        # Sample from the filtered lists
        sampled_treebank1 = random.sample(filtered_treebank1, min(max_samples, len(filtered_treebank1))) if filtered_treebank1 else []
        sampled_treebank2 = random.sample(filtered_treebank2, min(max_samples, len(filtered_treebank2))) if filtered_treebank2 else []
        sampled_treebank3 = random.sample(filtered_treebank3, min(max_samples, len(filtered_treebank3))) if filtered_treebank3 else []
        sampled_treebank4 = random.sample(filtered_treebank4, min(max_samples, len(filtered_treebank4))) if filtered_treebank4 else []
        sampled_treebank5 = random.sample(filtered_treebank5, min(max_samples, len(filtered_treebank5))) if filtered_treebank5 else []
        sampled_treebank6 = random.sample(filtered_treebank6, min(max_samples, len(filtered_treebank6))) if filtered_treebank6 else []
        
        # Compute distances to sampled treebank1
        distances1 = [tree_edit_distance(sentence, ref_sentence) for ref_sentence in sampled_treebank1]
        avg_distance1 = safe_average(distances1)
        
        # Compute distances to sampled treebank2
        distances2 = [tree_edit_distance(sentence, ref_sentence) for ref_sentence in sampled_treebank2]
        avg_distance2 = safe_average(distances2)
        
        # Compute distances to sampled treebank3
        distances3 = [tree_edit_distance(sentence, ref_sentence) for ref_sentence in sampled_treebank3]
        avg_distance3 = safe_average(distances3)
        
        # Compute distances to sampled treebank4
        distances4 = [tree_edit_distance(sentence, ref_sentence) for ref_sentence in sampled_treebank4]
        avg_distance4 = safe_average(distances4)
        
        # Compute distances to sampled treebank5
        distances5 = [tree_edit_distance(sentence, ref_sentence) for ref_sentence in sampled_treebank5]
        avg_distance5 = safe_average(distances5)
        
        # Compute distances to sampled treebank6
        distances6 = [tree_edit_distance(sentence, ref_sentence) for ref_sentence in sampled_treebank6]
        avg_distance6 = safe_average(distances6)
        
        results[i] = (avg_distance1, avg_distance2, avg_distance3, avg_distance4, avg_distance5, avg_distance6)
        
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare dependency tree structures')
    parser.add_argument('--output', help='Output file to write results', default='tree_distances.txt')
    
    args = parser.parse_args()
    
    # Hardcode the file paths
    input_file = 'data/yd-parsed.txt'
    #input_file = 'data/vedic.txt'
    treebank1_file = 'treebanks/la_proiel-ud-train.conllu'
    treebank2_file = 'treebanks/sa_vedic-ud-train.conllu'
    treebank3_file = 'treebanks/de_gsd-ud-train.conllu'
    treebank4_file = 'treebanks/lzh_kyoto-ud-train.conllu'
    treebank5_file = 'treebanks/is_icepahc-ud-train.conllu'
    treebank6_file = 'treebanks/sa_ufal-ud-test.conllu'
    
    # Read and parse the input files
    with open(input_file, 'r', encoding='utf-8') as f:
        input_text = f.read()
    with open(treebank1_file, 'r', encoding='utf-8') as f:
        treebank1_text = f.read()
    with open(treebank2_file, 'r', encoding='utf-8') as f:
        treebank2_text = f.read()
    with open(treebank3_file, 'r', encoding='utf-8') as f:
        treebank3_text = f.read()
    with open(treebank4_file, 'r', encoding='utf-8') as f:
        treebank4_text = f.read()
    with open(treebank5_file, 'r', encoding='utf-8') as f:
        treebank5_text = f.read()
    with open(treebank6_file, 'r', encoding='utf-8') as f:
        treebank6_text = f.read()
        
    # Parse the texts into dependency trees
    input_sentences = parse_custom_format(input_text)
    treebank1 = parse_ud_format(treebank1_text)
    treebank2 = parse_ud_format(treebank2_text)
    treebank3 = parse_ud_format(treebank3_text)
    treebank4 = parse_ud_format(treebank4_text)
    treebank5 = parse_ud_format(treebank5_text)
    treebank6 = parse_ud_format(treebank6_text)
    
    # Determine the length of the longest sentence in the input
    max_input_length = max(len(sentence) for sentence in input_sentences)
    
    # Prune sentences longer than the longest input sentence
    treebank1 = [sentence for sentence in treebank1 if len(sentence) <= max_input_length]
    treebank2 = [sentence for sentence in treebank2 if len(sentence) <= max_input_length]
    treebank3 = [sentence for sentence in treebank3 if len(sentence) <= max_input_length]
    treebank4 = [sentence for sentence in treebank4 if len(sentence) <= max_input_length]
    treebank5 = [sentence for sentence in treebank5 if len(sentence) <= max_input_length]
    treebank6 = [sentence for sentence in treebank6 if len(sentence) <= max_input_length]
    
    print(f"Parsed {len(input_sentences)} input sentences")
    print(f"Parsed {len(treebank1)} sentences from treebank 1 after pruning")
    print(f"Parsed {len(treebank2)} sentences from treebank 2 after pruning")
    print(f"Parsed {len(treebank3)} sentences from treebank 3 after pruning")
    print(f"Parsed {len(treebank4)} sentences from treebank 4 after pruning")
    print(f"Parsed {len(treebank5)} sentences from treebank 5 after pruning")
    print(f"Parsed {len(treebank6)} sentences from treebank 6 after pruning")
    
    # Compute the average distances and total distances
    results = compute_average_distances(input_sentences, treebank1, treebank2, treebank3, treebank4, treebank5, treebank6)
    
    # Initialize total distances
    total_distances = [0.0] * 6
    
    # Initialize counters for sentences closer to each treebank
    tb1_closer = tb2_closer = tb3_closer = tb4_closer = tb5_closer = tb6_closer = equal = 0

    # Write the results to the output file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("Sentence_ID\tAvg_Distance_Treebank1\tAvg_Distance_Treebank2\tAvg_Distance_Treebank3\tAvg_Distance_Treebank4\tAvg_Distance_Treebank5\tAvg_Distance_Treebank6\n")
        
        for sentence_id, (dist1, dist2, dist3, dist4, dist5, dist6) in sorted(results.items()):
            total_distances[0] += dist1
            total_distances[1] += dist2
            total_distances[2] += dist3
            total_distances[3] += dist4
            total_distances[4] += dist5
            total_distances[5] += dist6
            
            # Determine which treebank the sentence is closest to
            distances = [dist1, dist2, dist3, dist4, dist5, dist6]
            min_distance = min(distances)
            if distances.count(min_distance) > 1:
                equal += 1
            else:
                closest_index = distances.index(min_distance)
                if closest_index == 0:
                    tb1_closer += 1
                elif closest_index == 1:
                    tb2_closer += 1
                elif closest_index == 2:
                    tb3_closer += 1
                elif closest_index == 3:
                    tb4_closer += 1
                elif closest_index == 4:
                    tb5_closer += 1
                elif closest_index == 5:
                    tb6_closer += 1
            
            f.write(f"{sentence_id+1}\t{dist1:.4f}\t{dist2:.4f}\t{dist3:.4f}\t{dist4:.4f}\t{dist5:.4f}\t{dist6:.4f}\n")
    
    print(f"Results written to {args.output}")
    
    # Calculate mean and standard deviation of total distances
    mean_distance = np.mean(total_distances)
    std_distance = np.std(total_distances)
    
    # Print normalized distances as deviations from the mean
    print("Normalized Tree Edit Distances (Deviations from Mean):")
    print(f"- Deviation for Treebank1: {(total_distances[0] - mean_distance) / std_distance:.4f}")
    print(f"- Deviation for Treebank2: {(total_distances[1] - mean_distance) / std_distance:.4f}")
    print(f"- Deviation for Treebank3: {(total_distances[2] - mean_distance) / std_distance:.4f}")
    print(f"- Deviation for Treebank4: {(total_distances[3] - mean_distance) / std_distance:.4f}")
    print(f"- Deviation for Treebank5: {(total_distances[4] - mean_distance) / std_distance:.4f}")
    print(f"- Deviation for Treebank6: {(total_distances[5] - mean_distance) / std_distance:.4f}")
    
    def plot_summary(deviations, tb1_file, tb2_file, tb3_file, tb4_file, tb5_file, tb6_file):
        labels = [
            f'Treebank1\n({tb1_file})',
            f'Treebank2\n({tb2_file})',
            f'Treebank3\n({tb3_file})',
            f'Treebank4\n({tb4_file})',
            f'Treebank5\n({tb5_file})',
            f'Treebank6\n({tb6_file})',
        ]

        plt.figure(figsize=(12, 8))
        plt.bar(labels, deviations, color=['blue', 'orange', 'green', 'red', 'purple', 'brown'])
        plt.xlabel('Treebanks')
        plt.ylabel('Deviation from Mean')
        plt.title('Deviation of Total Distances from Mean for Each Treebank')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('deviation_result.png')
    
    # Calculate deviations
    deviations = [
        (total_distances[0] - mean_distance) / std_distance,
        (total_distances[1] - mean_distance) / std_distance,
        (total_distances[2] - mean_distance) / std_distance,
        (total_distances[3] - mean_distance) / std_distance,
        (total_distances[4] - mean_distance) / std_distance,
        (total_distances[5] - mean_distance) / std_distance
    ]

    # Call the plot_summary function with the computed deviations and filenames
    plot_summary(deviations, treebank1_file, treebank2_file, treebank3_file, treebank4_file, treebank5_file, treebank6_file)
    
if __name__ == "__main__":
    main()
