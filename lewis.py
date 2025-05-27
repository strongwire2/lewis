import re
from collections import Counter, defaultdict
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot
from itertools import permutations

# 옥텟 룰 기준 전자 수
ideal_valence_electrons = {
    'H': 2, 'He': 2, 'B': 6, 'Be': 4, 'C': 8, 'N': 8, 'O': 8, 'F': 8,
    'P': 8, 'S': 8, 'Cl': 8, 'Br': 8,
}

# 실제 원자가 전자 수 (free 상태)
valence_electrons = {
    'H': 1, 'He': 2, 'B': 3, 'Be': 2, 'C': 4, 'N': 5, 'O': 6, 'F': 7,
    'P': 5, 'S': 6, 'Cl': 7, 'Br': 7,
}


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
    """
    재귀를 돌면서 그래프의 노드와 엣지를 연결함.

    :param atoms: ['H_0', 'H_1', 'O_0']
    :param index: 몇번째 원소를 처리하고 있나
    :param graph: 그래프 객체
    :param result: 완성된 그래프를 담을 배열
    :return:
    """
    if index >= len(atoms):
        result.append(graph)
        return

    current_atom = atoms[index]
    if graph.nodes:
        print(f"attach {current_atom}")
        for node in graph.nodes:
            new_graph = graph.copy()
            new_graph.add_node(current_atom, label=current_atom.split('_')[0])
            new_graph.add_edge(node, current_atom, bond=1)
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


def verify_bond(graph):
    """
    Octet Rule을 만족하는지 검증하고, 필요하면 이중/삼중 결합으로 업데이트
    """
    for node in graph.nodes:
        print(graph.nodes[node].get('label'))


# https://dreampuf.github.io/GraphvizOnline  에서 확인

if __name__ == '__main__':
    #print(parse_formula("H2O"))
    #print(parse_formula("Cu2O4"))
    #print(parse_formula("O2Cu2O4"))
    #print(flatten_atom_counts(parse_formula("O2Cu2O4")))
    result = []
    traverse_lewis("H2O", result)
    for r in result:
        dot = to_pydot(r)
        print(dot.to_string())
        verify_bond(r)

    # print(permutate_atoms(['O_0', 'H_0', 'H_1']))
