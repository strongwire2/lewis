import networkx as nx


def find_longest_path(G):
    """
    nx graph의 longest path를 찾는다. SVG로 그리려면 longest path에 해당하는 노드를 먼저 가로로 배치해야 하기 때문
    longest path를 찾는 문제는 최적해가 없다고 한다. (NP Hard) 그래서 Brute-force를 쓸 수 밖에 없음.

    :param G:
    :return:
    """
    longest_path = []
    for source in G.nodes:
        for target in G.nodes:
            if source != target:
                for path in nx.all_simple_paths(G, source=source, target=target):
                    if len(path) > len(longest_path):
                        longest_path = path
    return longest_path


def test_find_longest_path():
    graph = nx.Graph()
    graph.add_node("O", label="O")
    graph.add_node("H1", label="H")
    graph.add_node("H2", label="H")
    graph.add_edge("H1", "O")
    graph.add_edge("H2", "O")
    path = find_longest_path(graph)
    print(path)

    graph2 = nx.Graph()
    graph2.add_edge("C_0", "H_0")
    graph2.add_edge("C_0", "H_1")
    graph2.add_edge("C_0", "H_2")
    graph2.add_edge("C_0", "H_3")
    path = find_longest_path(graph2)
    print(path)

if __name__ == '__main__':
    test_find_longest_path()

