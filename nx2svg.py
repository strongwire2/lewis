import networkx as nx
import drawsvg as draw


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


def add_dots(drawing, cx, cy, text_size, dot_count, loc):
    """
    비공유 전자를 그린다.

    :param drawing: drawSvg 객체
    :param cx: 텍스트 중심 좌표 x
    :param cy: 텍스트 중심 좌표 y
    :param text_size: 텍스트 크기
    :param dot_count: 점 수. 1 or 2
    :param loc: t(op), r(ight), b(ottom), l(eft)
    :return:
    """
    if loc == 't':
        # 윗 점
        drawing.append(draw.Circle(cx - 3, cy - text_size / 2 - 4, 2))
        if dot_count == 2:
            drawing.append(draw.Circle(cx + 3, cy - text_size / 2 - 4, 2))
    elif loc == 'r':
        # 오른쪽
        drawing.append(draw.Circle(cx + text_size / 2 + 1, cy - 2 - 3, 2))
        if dot_count == 2:
            drawing.append(draw.Circle(cx + text_size / 2 + 1, cy - 2 + 3, 2))
    elif loc == 'b':
        # 아래쪽
        drawing.append(draw.Circle(cx - 3, cy + text_size / 2 + 1, 2))
        if dot_count == 2:
            drawing.append(draw.Circle(cx + 3, cy + text_size / 2 + 1, 2))
    elif loc == 'l':
        # 왼쪽
        drawing.append(draw.Circle(cx - text_size / 2 - 1, cy - 2 - 3, 2))
        if dot_count == 2:
            drawing.append(draw.Circle(cx - text_size / 2 - 1, cy - 2 + 3, 2))


def test_svg():
    text_size = 24
    node_gap = 12

    d = draw.Drawing(100, 100)

    cx = 20
    cy = 30
    d.append(draw.Text("H", text_size, cx, cy, center=True, dominant_baseline='middle'))
    add_dots(d, cx, cy, text_size, 2, 't')
    add_dots(d, cx, cy, text_size, 2, 'r')
    add_dots(d, cx, cy, text_size, 2, 'b')
    add_dots(d, cx, cy, text_size, 2, 'l')
    d.save_svg('example.svg')
    #d.rasterize()  # Display as PNG


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
    test_svg()
