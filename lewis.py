import re
from collections import Counter, defaultdict
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot
from itertools import permutations

# 옥텟 룰 기준 전자 수
# TODO: 모든 원자에 대해 찾아서 추가할 것. 순서에 맞춰서
ideal_valence_electrons = {
    'H': 2, 'He': 2, 'B': 6, 'Be': 4, 'C': 8, 'N': 8, 'O': 8, 'F': 8,
    'P': 8, 'S': 8, 'Cl': 8, 'Br': 8,
}

# 실제 원자가 전자 수 (free 상태)
valence_electrons = {
    'H': 1, 'He': 2,
    'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8,
    'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 8,
    'K': 1, 'Ca': 2, 'Sc': 3, 'Ti': 4, 'V': 5, 'Cr': 6, 'Mn': 7, 'Fe': 8, 'Co': 9, 'Ni': 10, 'Cu': 11, 'Zn': 12,
    'Ga': 3, 'Ge': 4, 'As': 5, 'Se': 6, 'Br': 7, 'Kr': 8,
    'Rb': 1, 'Sr': 2, 'Y': 3, 'Zr': 4, 'Nb': 5, 'Mo': 6, 'Tc': 7, 'Ru': 8, 'Rh': 9, 'Pd': 10, 'Ag': 11, 'Cd': 12,
    'In': 3, 'Sn': 4, 'Sb': 5, 'Te': 6, 'I': 7, 'Xe': 8,
    'Cs': 1, 'Ba': 2, 'La': 3, 'Ce': 4, 'Pr': 5, 'Nd': 6, 'Pm': 7, 'Sm': 8, 'Eu': 9, 'Gd': 10, 'Tb': 11, 'Dy': 12, 'Ho': 13, 'Er': 14, 'Tm': 15, 'Yb': 16, 'Lu': 17,
    'Hf': 4, 'Ta': 5, 'W': 6, 'Re': 7, 'Os': 8, 'Ir': 9, 'Pt': 10, 'Au': 11, 'Hg': 12,
    'Tl': 3, 'Pb': 4, 'Bi': 5, 'Po': 6, 'At': 7, 'Rn': 8,
    'Fr': 1, 'Ra': 2, 'Ac': 3, 'Th': 4, 'Pa': 5, 'U': 6, 'Np': 7, 'Pu': 8, 'Am': 9, 'Cm': 10, 'Bk': 11, 'Cf': 12, 'Es': 13, 'Fm': 14, 'Md': 15, 'No': 16, 'Lr': 17,
    'Rf': 4, 'Db': 5, 'Sg': 6, 'Bh': 7, 'Hs': 8, 'Mt': 9, 'Ds': 10, 'Rg': 11, 'Cn': 12,
    'Nh': 3, 'Fl': 4, 'Mc': 5, 'Lv': 6, 'Ts': 7, 'Og': 8,
}

# 전기음성도
en_pauling_dict = {
    'H': 2.2, 'He': None,
    'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98, 'Ne': None,
    'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.9, 'P': 2.19, 'S': 2.58, 'Cl': 3.16, 'Ar': None,
    'K': 0.82, 'Ca': 1.0, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.9, 'Zn': 1.65,
    'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': None,
    'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 'Nb': 1.6, 'Mo': 2.16, 'Tc': 2.1, 'Ru': 2.2, 'Rh': 2.28, 'Pd': 2.2, 'Ag': 1.93, 'Cd': 1.69,
    'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.1, 'I': 2.66, 'Xe': 2.6,
    'Cs': 0.79, 'Ba': 0.89, 'La': 1.1, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14, 'Pm': None, 'Sm': 1.17, 'Eu': None, 'Gd': 1.2, 'Tb': None, 'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': None, 'Lu': 1.0,
    'Hf': 1.3, 'Ta': 1.5, 'W': 1.7, 'Re': 1.9, 'Os': 2.2, 'Ir': 2.2, 'Pt': 2.2, 'Au': 2.4, 'Hg': 1.9,
    'Tl': 1.8, 'Pb': 1.8, 'Bi': 1.9, 'Po': 2.0, 'At': 2.2, 'Rn': None,
    'Fr': 0.7, 'Ra': 0.9, 'Ac': 1.1, 'Th': 1.3, 'Pa': 1.5, 'U': 1.7, 'Np': 1.3, 'Pu': 1.3, 'Am': None, 'Cm': None, 'Bk': None, 'Cf': None, 'Es': None, 'Fm': None, 'Md': None, 'No': None, 'Lr': None,
    'Rf': None, 'Db': None, 'Sg': None, 'Bh': None, 'Hs': None, 'Mt': None, 'Ds': None, 'Rg': None, 'Cn': None,
    'Nh': None, 'Fl': None, 'Mc': None, 'Lv': None, 'Ts': None, 'Og': None,
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


def combine_atoms(atoms, index, graph, structs):
    """
    재귀를 돌면서 그래프의 노드와 엣지를 연결함.

    :param atoms: ['H_0', 'H_1', 'O_0']
    :param index: 몇번째 원소를 처리하고 있나
    :param graph: 그래프 객체
    :param structs: 완성된 그래프를 담을 배열
    :return:
    """
    if index >= len(atoms):
        structs.append(graph)
        return

    current_atom = atoms[index]
    if graph.nodes:
        #print(f"attach {current_atom}")
        for node in graph.nodes:
            new_graph = graph.copy()
            label = current_atom.split('_')[0]
            # 단일 결합으로 가정하고 추가되는 노드의 비공유전자 1개 줄임
            new_graph.add_node(current_atom, label=label, lone_e=valence_electrons.get(label, 0)-1)
            # 연결 추가
            new_graph.add_edge(node, current_atom, bond=1)
            # 추가되는 노드와 연결되는 노드의 비공유전자 1개 줄임
            new_graph.nodes[node]['lone_e'] = new_graph.nodes[node].get('lone_e', 0)-1
            #print(f"  add new {current_atom}, {node}-{current_atom}")
            combine_atoms(atoms, index+1, new_graph, structs)
    else:
        new_graph = graph.copy()
        label = current_atom.split('_')[0]
        new_graph.add_node(current_atom, label=label, lone_e=valence_electrons.get(label, 0))
        #print(f"add new {current_atom}")
        combine_atoms(atoms, index+1, new_graph, structs)


def traverse_lewis(formula):
    """
    화학식에서 원자를 분리한 다음, 순열(perms_list)을 만들고, combine_atoms를 호출하여
    :param formula: "H2O"와 같은 화학식
    :return: 모든 조합의 lewis 구조 Graph
    """
    structs = []
    atom_counts = parse_formula(formula)
    atoms = flatten_atom_counts(atom_counts)

    perms_list = permutate_atoms(atoms)

    for perms in perms_list:
        graph = nx.Graph()
        # print(f"atoms = {perms}")
        combine_atoms(perms, 0, graph, structs)
    return structs


def verify_bond(graph):
    """
    Octet Rule을 만족하는지 검증하고, 필요하면 이중/삼중 결합으로 업데이트

    :return True이면 Octet Rule 만족, False이면 만족하지 않음
    """
    for node in graph.nodes:
        #print(graph.nodes[node].get('label'))
        label = graph.nodes[node].get('label')  # 원소기호
        #print(graph.edges(node))
        ideal_valance = ideal_valence_electrons[label]  # 되어야 하는 원자가전자

        # 남아있는 비공유 전자 개수와 edge의 bond 수를 합쳐서 Octet Rule을 만족하는지 확인
        e_count = graph.nodes[node].get('lone_e')
        edges = list(graph.edges(node, data=True))
        for u, v, data in edges:
            # print(f"{u} -- {v}, data: {data}")
            e_count += data.get("bond", 1)*2  # 공유 전자쌍이므로 연결당 2개의 전자임.
        # Octet Rule이 안맞으면 False 리턴
        if e_count != ideal_valance:
            return False

    # 모든 원소에 대해 Octet Rule을 만족하므로 True 리턴
    return True


def get_lewis_struct(formular):
    """
    주어진 화학식으로 lewis 구조를 만들고, Octet Rule이 만족하는 구조를 Graph의 list로 리턴
    :param formular: 화학식 "H2O"의 형식
    :return: Octet Rule을 만족하는 Graph의 list
    """
    structs = traverse_lewis(formular)
    verified = []
    for r in structs:
        if verify_bond(r):
            verified.append(r)
    return verified


# https://dreampuf.github.io/GraphvizOnline  에서 확인

if __name__ == '__main__':
    #print(parse_formula("H2O"))
    #print(parse_formula("Cu2O4"))
    #print(parse_formula("O2Cu2O4"))
    #print(flatten_atom_counts(parse_formula("O2Cu2O4")))
    result = get_lewis_struct("H2O")
    for r in result:
        dot = to_pydot(r)
        print(dot.to_string())


