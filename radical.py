import re
from collections import Counter, defaultdict
import networkx as nx
import pydot
from networkx.drawing.nx_pydot import to_pydot, from_pydot
from itertools import permutations

# 확장 옥텟 및 기본 옥텟 리스트형으로 선언
ideal_valence_electrons = {
    # 기본 옥텟
    'H': [2], 'He': [2], 'B': [6], 'Be': [4],
    'C': [8], 'N': [8], 'O': [8], 'F': [8],
    # 확장 옥텟 허용 (10, 12 등 추가)
    'P': [8, 10], 'S': [8, 10, 12], 'Cl': [8, 10, 12], 'Br': [8, 10, 12], 'I': [8, 10, 12],
    'Xe': [8, 10, 12], 'Kr': [8, 10, 12],
    # 나머지는 [8]
}

def get_allowed_valence(symbol):
    return ideal_valence_electrons.get(symbol, [8])

# 라디칼 허용 원소 (예: NO, NO2, ClO2, O2 등)
allowed_radicals = {'N', 'O', 'Cl', 'Br', 'I', 'S'}

# 실제 원자가 전자 수 (free 상태)
valence_electrons = {
    'H': 1, 'He': 2, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8,
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

def parse_formula(formula_str):
    tokens = re.findall(r'([A-Z][a-z]*)(\d*)', formula_str)
    atom_counts = Counter()
    for symbol, count_str in tokens:
        count = int(count_str) if count_str else 1
        atom_counts[symbol] += count
    return atom_counts

def flatten_atom_counts(atom_counts):
    flat_atoms = []
    for symbol, count in atom_counts.items():
        for i in range(count):
            flat_atoms.append(f"{symbol}_{i}")
    return flat_atoms

def permutate_atoms(atoms):
    elements = [a.split('_')[0] for a in atoms]
    element_ids = defaultdict(list)
    for atom in atoms:
        symbol, index = atom.split('_')
        element_ids[symbol].append(atom)
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
    if index >= len(atoms):
        structs.append(graph)
        return
    current_atom = atoms[index]
    if graph.nodes:
        for node in graph.nodes:
            new_graph = graph.copy()
            label = current_atom.split('_')[0]
            new_graph.add_node(current_atom, label=label, lone_e=valence_electrons.get(label, 0)-1)
            new_graph.add_edge(node, current_atom, bond=1)
            new_graph.nodes[node]['lone_e'] = new_graph.nodes[node].get('lone_e', 0)-1
            if pruning(new_graph):
                continue
            combine_atoms(atoms, index+1, new_graph, structs)
    else:
        new_graph = graph.copy()
        label = current_atom.split('_')[0]
        new_graph.add_node(current_atom, label=label, lone_e=valence_electrons.get(label, 0))
        combine_atoms(atoms, index+1, new_graph, structs)

def traverse_lewis(formula):
    structs = []
    atom_counts = parse_formula(formula)
    atoms = flatten_atom_counts(atom_counts)
    perms_list = permutate_atoms(atoms)
    for perms in perms_list:
        graph = nx.Graph()
        combine_atoms(perms, 0, graph, structs)
    return structs

def pruning(graph):
    for node in graph.nodes:
        if graph.nodes[node].get('lone_e', 0) < 0:
            return True

def verify_bond(graph):
    # 옥텟/확장옥텟/라디칼 동시 판별
    for node in graph.nodes:
        label = graph.nodes[node].get('label')
        allowed_vals = get_allowed_valence(label)
        while True:
            e_count = graph.nodes[node].get('lone_e', 0)
            edges = list(graph.edges(node, data=True))
            for u, v, data in edges:
                e_count += data.get("bond", 1)*2
            # 확장 옥텟: 여러 개 중 하나라도 맞으면 break
            if e_count in allowed_vals:
                break
            # 라디칼 케이스 (allowed_radicals에 포함, 비공유전자 1개 부족)
            if (label in allowed_radicals) and (e_count == min(allowed_vals)-1):
                graph.nodes[node]['radical'] = True
                break
            something_changed = False
            for u, v, data in edges:
                other = v if u == node else u
                if graph.nodes[other].get('lone_e', 0) > 0 and graph.nodes[node].get('lone_e', 0) > 0:
                    graph.nodes[other]['lone_e'] -= 1
                    graph.nodes[node]['lone_e'] -= 1
                    graph[u][v]['bond'] += 1
                    something_changed = True
                    break
            if not something_changed:
                return False
    # 마지막 검증
    for node in graph.nodes:
        label = graph.nodes[node].get('label')
        allowed_vals = get_allowed_valence(label)
        e_count = graph.nodes[node].get('lone_e', 0)
        for u, v, data in graph.edges(node, data=True):
            e_count += data.get("bond", 1) * 2
        if not (e_count in allowed_vals or (label in allowed_radicals and e_count == min(allowed_vals)-1)):
            return False
    return True

def get_lewis_struct(formula):
    structs = traverse_lewis(formula)
    verified = []
    for g in structs:
        if verify_bond(g):
            verified.append(g)
    return verified

def annotate_lewis(graph):
    for node in graph.nodes:
        label_txt = graph.nodes[node].get('label')
        lone_e = graph.nodes[node].get('lone_e', 0)
        radical = graph.nodes[node].get('radical', False)
        label_str = f"{label_txt}:{lone_e}"
        if radical:
            label_str += "•"  # 라디칼 표시
        graph.nodes[node]['label'] = label_str
    for u, v, data in graph.edges(data=True):
        if data.get("bond") > 1:
            graph[u][v]['label'] = data.get("bond")

def are_graphs_equivalent(g1, g2):
    def get_node_labels(graph):
        return sorted([data["label"] for _, data in graph.nodes(data=True)])
    if get_node_labels(g1) != get_node_labels(g2):
        return False
    def get_edge_signatures(graph):
        sigs = []
        for u, v, data in graph.edges(data=True):
            label_u = graph.nodes[u]["label"]
            label_v = graph.nodes[v]["label"]
            bond = data.get("bond", 1)
            sig = tuple(sorted([label_u, label_v]) + [bond])
            sigs.append(sig)
        return sorted(sigs)
    return get_edge_signatures(g1) == get_edge_signatures(g2)

def formal_charge(graph):
    fc_dict = {}
    for node in graph.nodes:
        label = graph.nodes[node].get('label').split(":")[0].replace("•", "")
        V = valence_electrons[label]
        N = graph.nodes[node].get('lone_e', 0)
        edges = list(graph.edges(node, data=True))
        bond_count = 0
        for u, v, data in edges:
            bond_count += data.get("bond", 1)
        B = bond_count
        fc = V - (N + B)
        fc_dict[node] = fc
    return fc_dict

def test_fail_case():
    G = nx.Graph()
    G.add_node("O_0", label="O", lone_e=4)
    G.add_node("C_0", label="C", lone_e=3)
    G.add_node("O_1", label="O", lone_e=5)
    G.add_edge("O_0", "C_0", bond=1)
    G.add_edge("O_0", "O_1", bond=1)
    print(verify_bond(G))
    dot = to_pydot(G)
    print(dot.to_string())


if __name__ == '__main__':
    # 예시 사용
    # result = get_lewis_struct("NO2")   # 라디칼
    # result = get_lewis_struct("O3")    # 라디칼 (확장 옥텟은 X)
    # result = get_lewis_struct("SF4")   # 확장 옥텟
    result = get_lewis_struct("CH3COOH")     # 확장 옥텟 케이스
    for r in result:
        annotate_lewis(r)
        dot = to_pydot(r)
        print(dot.to_string())
        print("Formal charges:", formal_charge(r))