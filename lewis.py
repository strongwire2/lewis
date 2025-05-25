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
    if index >= len(atoms):
        result.append(graph)
        return

    current_atom = atoms[index]
    if graph.nodes:
        print(f"attach {current_atom}")
        for node in graph.nodes:
            new_graph = graph.copy()
            new_graph.add_node(current_atom)
            new_graph.add_edge(node, current_atom)
            print(f"  add new {current_atom}, {node}-{current_atom}")
            combine_atoms(atoms, index+1, new_graph, result)
    else:
        new_graph = graph.copy()
        new_graph.add_node(current_atom)
        print(f"add new {current_atom}")
        combine_atoms(atoms, index+1, new_graph, result)


def traverse_lewis(formula, result):
    atom_counts = parse_formula(formula)
    atoms = flatten_atom_counts(atom_counts)
    graph = nx.Graph()
    print(f"atoms = {atoms}")
    combine_atoms(atoms, 0, graph, result)



if __name__ == '__main__':
    #print(parse_formula("H2O"))
    #print(parse_formula("Cu2O4"))
    #print(parse_formula("O2Cu2O4"))
    #print(flatten_atom_counts(parse_formula("O2Cu2O4")))
    result = []
    traverse_lewis("OH2", result)
    for r in result:
        dot = to_pydot(r)
        print(dot.to_string())
