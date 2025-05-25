import re
from collections import Counter
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot


def parse_formula(formula_str):
    """
    간단한 화학식 파서 (예: H2O, NH3)
    예를 들어 H2O를 넣으면  {'H': 2, 'O': 1}
    """
    tokens = re.findall(r'([A-Z][a-z]*)(\d*)', formula_str)
    atom_counts = Counter()  # Counter는 symbol 이 몇개 있나를 저장
    for symbol, count_str in tokens:
        count = int(count_str) if count_str else 1
        atom_counts[symbol] += count
    return atom_counts


def flatten_atom_counts(atom_counts):
    """
    {'H': 2, 'O': 1} → ['H_0', 'H_1', 'O_0']
    """
    flat_atoms = []
    for symbol, count in atom_counts.items():
        for i in range(count):
            flat_atoms.append(f"{symbol}_{i}")
    return flat_atoms


def combine_atoms(atoms, index, graph, result):
    current_atom = atoms[index]
    if graph.nodes:
        for node in graph.nodes:
            pass
    else:
        graph.add_node(current_atom)


def traverse_lewis(formula):
    atom_counts = parse_formula(formula)
    atoms = flatten_atom_counts(atom_counts)
    result = []
    graph = nx.Graph()

    combine_atoms(atoms, 0, graph, result)

if __name__ == '__main__':
    print(parse_formula("H2O"))
    print(parse_formula("Cu2O4"))
    print(parse_formula("O2Cu2O4"))
    print(flatten_atom_counts(parse_formula("O2Cu2O4")))
