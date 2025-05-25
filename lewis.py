import re
from collections import Counter, defaultdict
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot
from itertools import permutations


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


def permutate_atoms(atoms):
    """
    ['O_0', 'H_0', 'H_1'] 이런 형식을 받아서 순열로 배치한 다음 중복을 제거하고 다시 번호를 붙임.

    ('H', 'H', 'O') → ['H_0', 'H_1', 'O_0']
    ('H', 'O', 'H') → ['H_0', 'O_0', 'H_1']
    ('O', 'H', 'H') → ['O_0', 'H_0', 'H_1']
    """
    # 1. 원소 이름만 추출
    elements = [a.split('_')[0] for a in atoms]  # ['O', 'H', 'H']
    # 2. 원소 ID 목록을 그룹화 (예: {'O': ['O_0'], 'H': ['H_0', 'H_1']})
    element_ids = defaultdict(list)  # defaultdict는 해당하는 key가 없을때 자동으로 초기화된 값을 리턴함
    for atom in atoms:
        symbol, index = atom.split('_')
        element_ids[symbol].append(atom)
    # 3. 가능한 순열 구하기 (중복 포함)
    perms = set(permutations(elements))

    perms_list = []
    for p in sorted(perms):
        counters = defaultdict(int)
        result = []
        for symbol in p:
            result.append(element_ids[symbol][counters[symbol]])
            counters[symbol] += 1
        perms_list.append(result)

    return perms_list


def combine_atoms(atoms, index, graph, result):
    if index >= len(atoms):
        result.append(graph)
        return

    current_atom = atoms[index]
    if graph.nodes:
        print(f"attach {current_atom}")
        for node in graph.nodes:
            new_graph = graph.copy()
            new_graph.add_node(current_atom, label=current_atom.split('_')[0])
            new_graph.add_edge(node, current_atom)
            print(f"  add new {current_atom}, {node}-{current_atom}")
            combine_atoms(atoms, index+1, new_graph, result)
    else:
        new_graph = graph.copy()
        new_graph.add_node(current_atom, label=current_atom.split('_')[0])
        print(f"add new {current_atom}")
        combine_atoms(atoms, index+1, new_graph, result)


def traverse_lewis(formula, result):
    atom_counts = parse_formula(formula)
    atoms = flatten_atom_counts(atom_counts)

    perms_list = permutate_atoms(atoms)

    for perms in perms_list:
        graph = nx.Graph()
        print(f"atoms = {perms}")
        combine_atoms(perms, 0, graph, result)


# https://dreampuf.github.io/GraphvizOnline  에서 확인

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

    # print(permutate_atoms(['O_0', 'H_0', 'H_1']))
