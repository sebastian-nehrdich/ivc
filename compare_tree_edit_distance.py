import re
import numpy as np
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Any, Optional
import matplotlib.pyplot as plt


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
    
    return total_distance / max_size


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
    
    for i, sentence in enumerate(input_sentences):
        # Compute distances to treebank1
        distances1 = [tree_edit_distance(sentence, ref_sentence) for ref_sentence in treebank1]
        avg_distance1 = np.mean(distances1) if distances1 else float('inf')
        
        # Compute distances to treebank2
        distances2 = [tree_edit_distance(sentence, ref_sentence) for ref_sentence in treebank2]
        avg_distance2 = np.mean(distances2) if distances2 else float('inf')
        
        # Compute distances to treebank3
        distances3 = [tree_edit_distance(sentence, ref_sentence) for ref_sentence in treebank3]
        avg_distance3 = np.mean(distances3) if distances3 else float('inf')
        
        # Compute distances to treebank4
        distances4 = [tree_edit_distance(sentence, ref_sentence) for ref_sentence in treebank4]
        avg_distance4 = np.mean(distances4) if distances4 else float('inf')
        
        # Compute distances to treebank5
        distances5 = [tree_edit_distance(sentence, ref_sentence) for ref_sentence in treebank5]
        avg_distance5 = np.mean(distances5) if distances5 else float('inf')
        
        # Compute distances to treebank6
        distances6 = [tree_edit_distance(sentence, ref_sentence) for ref_sentence in treebank6]
        avg_distance6 = np.mean(distances6) if distances6 else float('inf')
        
        results[i] = (avg_distance1, avg_distance2, avg_distance3, avg_distance4, avg_distance5, avg_distance6)
        
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare dependency tree structures')
    parser.add_argument('--output', help='Output file to write results', default='tree_distances.txt')
    
    args = parser.parse_args()
    
    # Hardcode the file paths
    input_file = 'data/yd-parsed.txt'
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
    
    # Compute the average distances
    results = compute_average_distances(input_sentences, treebank1, treebank2, treebank3, treebank4, treebank5, treebank6)
    
    # Write the results to the output file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("Sentence_ID\tAvg_Distance_Treebank1\tAvg_Distance_Treebank2\tAvg_Distance_Treebank3\tAvg_Distance_Treebank4\tAvg_Distance_Treebank5\tAvg_Distance_Treebank6\tCloser_To\tTreebank_File\n")
        
        for sentence_id, (dist1, dist2, dist3, dist4, dist5, dist6) in sorted(results.items()):
            if dist1 < dist2 and dist1 < dist3 and dist1 < dist4 and dist1 < dist5 and dist1 < dist6:
                closer_to = "Treebank1"
                treebank_file = treebank1_file
            elif dist2 < dist1 and dist2 < dist3 and dist2 < dist4 and dist2 < dist5 and dist2 < dist6:
                closer_to = "Treebank2"
                treebank_file = treebank2_file
            elif dist3 < dist1 and dist3 < dist2 and dist3 < dist4 and dist3 < dist5 and dist3 < dist6:
                closer_to = "Treebank3"
                treebank_file = treebank3_file
            elif dist4 < dist1 and dist4 < dist2 and dist4 < dist3 and dist4 < dist5 and dist4 < dist6:
                closer_to = "Treebank4"
                treebank_file = treebank4_file
            elif dist5 < dist1 and dist5 < dist2 and dist5 < dist3 and dist5 < dist4 and dist5 < dist6:
                closer_to = "Treebank5"
                treebank_file = treebank5_file
            elif dist6 < dist1 and dist6 < dist2 and dist6 < dist3 and dist6 < dist4 and dist6 < dist5:
                closer_to = "Treebank6"
                treebank_file = treebank6_file
            else:
                closer_to = "Equal"
                treebank_file = "Multiple"
                
            f.write(f"{sentence_id+1}\t{dist1:.4f}\t{dist2:.4f}\t{dist3:.4f}\t{dist4:.4f}\t{dist5:.4f}\t{dist6:.4f}\t{closer_to}\t{treebank_file}\n")
    
    print(f"Results written to {args.output}")
    
    # Print summary statistics
    tb1_closer = sum(1 for _, (d1, d2, d3, d4, d5, d6) in results.items() if d1 < d2 and d1 < d3 and d1 < d4 and d1 < d5 and d1 < d6)
    tb2_closer = sum(1 for _, (d1, d2, d3, d4, d5, d6) in results.items() if d2 < d1 and d2 < d3 and d2 < d4 and d2 < d5 and d2 < d6)
    tb3_closer = sum(1 for _, (d1, d2, d3, d4, d5, d6) in results.items() if d3 < d1 and d3 < d2 and d3 < d4 and d3 < d5 and d3 < d6)
    tb4_closer = sum(1 for _, (d1, d2, d3, d4, d5, d6) in results.items() if d4 < d1 and d4 < d2 and d4 < d3 and d4 < d5 and d4 < d6)
    tb5_closer = sum(1 for _, (d1, d2, d3, d4, d5, d6) in results.items() if d5 < d1 and d5 < d2 and d5 < d3 and d5 < d4 and d5 < d6)
    tb6_closer = sum(1 for _, (d1, d2, d3, d4, d5, d6) in results.items() if d6 < d1 and d6 < d2 and d6 < d3 and d6 < d4 and d6 < d5)
    equal = sum(1 for _, (d1, d2, d3, d4, d5, d6) in results.items() if (d1 == d2 and d1 < d3 and d1 < d4 and d1 < d5 and d1 < d6) or (d1 == d3 and d1 < d2 and d1 < d4 and d1 < d5 and d1 < d6) or (d1 == d4 and d1 < d2 and d1 < d3 and d1 < d5 and d1 < d6) or (d1 == d5 and d1 < d2 and d1 < d3 and d1 < d4 and d1 < d6) or (d1 == d6 and d1 < d2 and d1 < d3 and d1 < d4 and d1 < d5) or (d2 == d3 and d2 < d1 and d2 < d4 and d2 < d5 and d2 < d6) or (d2 == d4 and d2 < d1 and d2 < d3 and d2 < d5 and d2 < d6) or (d2 == d5 and d2 < d1 and d2 < d3 and d2 < d4 and d2 < d6) or (d2 == d6 and d2 < d1 and d2 < d3 and d2 < d4 and d2 < d5) or (d3 == d4 and d3 < d1 and d3 < d2 and d3 < d5 and d3 < d6) or (d3 == d5 and d3 < d1 and d3 < d2 and d3 < d4 and d3 < d6) or (d3 == d6 and d3 < d1 and d3 < d2 and d3 < d4 and d3 < d5) or (d4 == d5 and d4 < d1 and d4 < d2 and d4 < d3 and d4 < d6) or (d4 == d6 and d4 < d1 and d4 < d2 and d4 < d3 and d4 < d5) or (d5 == d6 and d5 < d1 and d5 < d2 and d5 < d3 and d5 < d4) or (d1 == d2 == d3 and d1 < d4 and d1 < d5 and d1 < d6) or (d1 == d2 == d4 and d1 < d3 and d1 < d5 and d1 < d6) or (d1 == d2 == d5 and d1 < d3 and d1 < d4 and d1 < d6) or (d1 == d2 == d6 and d1 < d3 and d1 < d4 and d1 < d5) or (d1 == d3 == d4 and d1 < d2 and d1 < d5 and d1 < d6) or (d1 == d3 == d5 and d1 < d2 and d1 < d4 and d1 < d6) or (d1 == d3 == d6 and d1 < d2 and d1 < d4 and d1 < d5) or (d1 == d4 == d5 and d1 < d2 and d1 < d3 and d1 < d6) or (d1 == d4 == d6 and d1 < d2 and d1 < d3 and d1 < d5) or (d1 == d5 == d6 and d1 < d2 and d1 < d3 and d1 < d4) or (d2 == d3 == d4 and d2 < d1 and d2 < d5 and d2 < d6) or (d2 == d3 == d5 and d2 < d1 and d2 < d4 and d2 < d6) or (d2 == d3 == d6 and d2 < d1 and d2 < d4 and d2 < d5) or (d2 == d4 == d5 and d2 < d1 and d2 < d3 and d2 < d6) or (d2 == d4 == d6 and d2 < d1 and d2 < d3 and d2 < d5) or (d2 == d5 == d6 and d2 < d1 and d2 < d3 and d2 < d4) or (d3 == d4 == d5 and d3 < d1 and d3 < d2 and d3 < d6) or (d3 == d4 == d6 and d3 < d1 and d3 < d2 and d3 < d5) or (d3 == d5 == d6 and d3 < d1 and d3 < d2 and d3 < d4) or (d4 == d5 == d6 and d4 < d1 and d4 < d2 and d4 < d3) or (d1 == d2 == d3 == d4 and d1 < d5 and d1 < d6) or (d1 == d2 == d3 == d5 and d1 < d4 and d1 < d6) or (d1 == d2 == d3 == d6 and d1 < d4 and d1 < d5) or (d1 == d2 == d4 == d5 and d1 < d3 and d1 < d6) or (d1 == d2 == d4 == d6 and d1 < d3 and d1 < d5) or (d1 == d2 == d5 == d6 and d1 < d3 and d1 < d4) or (d1 == d3 == d4 == d5 and d1 < d2 and d1 < d6) or (d1 == d3 == d4 == d6 and d1 < d2 and d1 < d5) or (d1 == d3 == d5 == d6 and d1 < d2 and d1 < d4) or (d1 == d4 == d5 == d6 and d1 < d2 and d1 < d3) or (d2 == d3 == d4 == d5 and d2 < d1 and d2 < d6) or (d2 == d3 == d4 == d6 and d2 < d1 and d2 < d5) or (d2 == d3 == d5 == d6 and d2 < d1 and d2 < d4) or (d2 == d4 == d5 == d6 and d2 < d1 and d2 < d3) or (d3 == d4 == d5 == d6 and d3 < d1 and d3 < d2) or (d1 == d2 == d3 == d4 == d5 and d1 < d6) or (d1 == d2 == d3 == d4 == d6 and d1 < d5) or (d1 == d2 == d3 == d5 == d6 and d1 < d4) or (d1 == d2 == d4 == d5 == d6 and d1 < d3) or (d1 == d3 == d4 == d5 == d6 and d1 < d2) or (d2 == d3 == d4 == d5 == d6 and d2 < d1) or (d1 == d2 == d3 == d4 == d5 == d6))
    
    print(f"Summary:")
    print(f"- Sentences closer to Treebank1 ({treebank1_file}): {tb1_closer} ({tb1_closer/len(results)*100:.1f}%)")
    print(f"- Sentences closer to Treebank2 ({treebank2_file}): {tb2_closer} ({tb2_closer/len(results)*100:.1f}%)")
    print(f"- Sentences closer to Treebank3 ({treebank3_file}): {tb3_closer} ({tb3_closer/len(results)*100:.1f}%)")
    print(f"- Sentences closer to Treebank4 ({treebank4_file}): {tb4_closer} ({tb4_closer/len(results)*100:.1f}%)")
    print(f"- Sentences closer to Treebank5 ({treebank5_file}): {tb5_closer} ({tb5_closer/len(results)*100:.1f}%)")
    print(f"- Sentences closer to Treebank6 ({treebank6_file}): {tb6_closer} ({tb6_closer/len(results)*100:.1f}%)")
    print(f"- Sentences equally distant: {equal} ({equal/len(results)*100:.1f}%)")
    
    def plot_summary(tb1_closer, tb2_closer, tb3_closer, tb4_closer, tb5_closer, tb6_closer, equal, total,
                     tb1_file, tb2_file, tb3_file, tb4_file, tb5_file, tb6_file):
        labels = [
            f'Treebank1\n({tb1_file})',
            f'Treebank2\n({tb2_file})',
            f'Treebank3\n({tb3_file})',
            f'Treebank4\n({tb4_file})',
            f'Treebank5\n({tb5_file})',
            f'Treebank6\n({tb6_file})',
            'Equal'
        ]
        counts = [tb1_closer, tb2_closer, tb3_closer, tb4_closer, tb5_closer, tb6_closer, equal]
        percentages = [count / total * 100 for count in counts]

        plt.figure(figsize=(12, 8))
        plt.bar(labels, percentages, color=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'gray'])
        plt.xlabel('Treebanks')
        plt.ylabel('Percentage of Sentences (%)')
        plt.title('Summary of Sentences Closer to Each Treebank')
        plt.ylim(0, 100)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # Call the plot_summary function with the computed values and filenames
    plot_summary(tb1_closer, tb2_closer, tb3_closer, tb4_closer, tb5_closer, tb6_closer, equal, len(results),
                 treebank1_file, treebank2_file, treebank3_file, treebank4_file, treebank5_file, treebank6_file)
    
if __name__ == "__main__":
    main()
