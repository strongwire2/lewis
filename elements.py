from mendeleev import element, get_all_elements
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt

def test_elements():
    """
    모든 원소에 대해 원자번호(전자수), 원자가전자수, 전기음성도 출력
    :return:
    """
    # elements_to_check = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Na', 'Mg', 'Al', 'Si', 'He', 'Ar']

    for el in get_all_elements():
        # el = element(e)
        # nvalence 원자가전자, en_pauling 전기음성도
        print(f"{el.symbol}: {el.atomic_number}, {el.nvalence()}, {el.en_pauling}")


def print_elements_data():
    """
    원자가전자와 전기음성도에 대한 dict 출력. lewis.py 에서 사용
    :return:
    """
    nvalence_dict = {}
    en_pauling_dict = {}

    for el in get_all_elements():
        try:
            nval = el.nvalence()
        except Exception:
            nval = None
        en = el.en_pauling

        nvalence_dict[el.symbol] = nval
        en_pauling_dict[el.symbol] = en

    # 결과 출력
    print("nvalence_dict = {")
    for k, v in nvalence_dict.items():
        print(f"    '{k}': {v},")
    print("}")

    print("\nen_pauling_dict = {")
    for k, v in en_pauling_dict.items():
        print(f"    '{k}': {v},")
    print("}")


def test_network():
    G = nx.Graph()
    G.add_node("X", label="Start", color="red")
    G.add_node("K")
    G.add_edge("X", "A") #없는 노드와 edge를 연결하면 노드를 만듦.
    G.add_edge("X", "K")
    pos = nx.spring_layout(G)  # 또는
    # pos = nx.kamada_kawai_layout(G)  # ,
    # pos = nx.circular_layout(G)

    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=1500, font_size=12)

    # 엣지 라벨이 있다면 표시
    edge_labels = nx.get_edge_attributes(G, 'label')
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Network Structure")
    plt.axis('off')
    plt.show()



if __name__ == '__main__':
    # test_elements()
    # test_network()
    print_elements_data()