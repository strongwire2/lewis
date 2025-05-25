from mendeleev import element
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt

def test_elements():
    elements_to_check = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Na', 'Mg', 'Al', 'Si', 'He', 'Ar']

    for e in elements_to_check:
        el = element(e)
        # nvalence 원자가전자, en_pauling 전기음성도
        print(f"{el.symbol}: {el.atomic_number}, {el.nvalence()}, {el.en_pauling}")


def test_network():
    G = nx.Graph()
    G.add_node("X", label="Start", color="red")
    G.add_node("K")
    G.add_edge("X", "A")
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
    test_elements()
    test_network()