import glob
import os
import matplotlib.pyplot as plt
import numpy as np
# ... existing code ...

class DependencyNode:
    def __init__(self, id, form, head, deprel=None):
        self.id = id         # integer ID of the token in the sentence, usually 1..N
        self.form = form     # string form of the token (e.g., the surface word)
        self.head = head     # integer ID of the head of this token (0 if root)
        self.deprel = deprel # dependency relation label (optional)
        self.children = []   # list of child DependencyNodes

def validate_tree(nodes):
    """
    Validate a list of DependencyNode objects representing one sentence.
    
    Returns True if the sentence is a valid dependency tree, False otherwise.
    """
    num_tokens = len(nodes)
    if num_tokens == 0:
        return False
    
    # Step 1: Check root count
    roots = [node for node in nodes if node.head == 0]
    if len(roots) != 1:
        return False

    # Step 2: Check valid heads
    for node in nodes:
        if node.head != 0:
            # Must be 1..N
            if not (1 <= node.head <= num_tokens):
                return False
            # Head cannot be the same ID
            if node.head == node.id:
                return False
    
    # Step 3: Build adjacency list
    adjacency_list = {node.id: [] for node in nodes}
    for node in nodes:
        if node.head != 0:
            # Ensure the head is a valid node ID
            if node.head in adjacency_list:
                adjacency_list[node.head].append(node.id)
            else:
                return False  # Invalid head reference
    
    # 4) Detect cycles and check connectivity using DFS or BFS from the root.
    root_id = roots[0].id
    
    visited = set()
    in_stack = set()  # to detect back-edges

    def dfs(node_id):
        """ Returns True if a cycle is found, otherwise False. """
        visited.add(node_id)
        in_stack.add(node_id)
        
        for child_id in adjacency_list[node_id]:
            if child_id not in visited:
                if dfs(child_id):
                    return True  # cycle found in a descendant
            elif child_id in in_stack:
                # child_id is in the current recursion stack => cycle
                return True
        
        in_stack.remove(node_id)
        return False

    # Run DFS from the single root
    if dfs(root_id):
        return False  # a cycle was found
    
    # 5) Check that all tokens were visited => single connected component.
    if len(visited) != num_tokens:
        return False
    
    # 6) Check for >50% "fla" labels
    fla_count = sum(1 for node in nodes if node.deprel == 'fla')
    if fla_count > num_tokens / 2:
        return False
    
    return True

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
        
        sentences.append(current_sentence)  # Moved outside the inner loop
    return sentences

def main():
    parsed_files = glob.glob('data/*-parsed.txt')
    if not parsed_files:
        print("No parsed files found.")
        return

    error_rates = {}
    fla_counts = {}  # Dictionary to store the count of sentences with >50% "fla" labels per file

    for file in parsed_files:
        with open(file, 'r', encoding='utf-8') as f:
            sentences = parse_custom_format(f.read())
        if not sentences:
            print(f"No sentences found in {file}.")
            continue
        
        valid_trees = []
        invalid_sentences = []
        fla_sentences_count = 0  # Counter for sentences with >50% "fla" labels
        
        for sentence in sentences:
            if validate_tree(sentence):
                valid_trees.append(sentence)
            else:
                invalid_sentences.append(sentence)
            
            # Check for >50% "fla" labels
            fla_count = sum(1 for node in sentence if node.deprel == 'fla')
            if fla_count > len(sentence) / 2:
                fla_sentences_count += 1
        
        error_rate = 1 - len(valid_trees) / len(sentences) if len(sentences) > 0 else 0
        error_rates[os.path.basename(file)] = error_rate
        
        # Calculate the percentage of sentences with >50% "fla" labels
        fla_percentage = (fla_sentences_count / len(sentences)) * 100 if len(sentences) > 0 else 0
        fla_counts[os.path.basename(file)] = fla_percentage

        # Print invalid sentences with their trees
        if invalid_sentences:
            print(f"\nInvalid sentences in {file}:")
            for sentence in invalid_sentences:
                print("Sentence:", " ".join(node.form for node in sentence))
                print("Dependency Tree:")
                for node in sentence:
                    print(f"ID: {node.id}, Form: {node.form}, Head: {node.head}, Deprel: {node.deprel}")
                print()  # Add a newline for better readability

    # Plot the error rates
    plt.figure(figsize=(12, 8))
    labels, rates = zip(*error_rates.items())
    bars = plt.bar(labels, rates, color='salmon')

    # Add data labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2%}', va='bottom', ha='center')

    plt.ylabel('Error Rate')
    plt.title('Error Rate of Dependency Trees in Parsed Files')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('error_rates.png')

    # Print error rates and fla counts to the console
    print("\nError Rates:")
    for filename, rate in error_rates.items():
        print(f"{filename}: {rate:.2%}")

    print("\nSentences with >50% 'fla' labels:")
    for filename, percentage in fla_counts.items():
        print(f"{filename}: {percentage:.2f}%")

if __name__ == "__main__":
    main()