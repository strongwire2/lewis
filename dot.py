import networkx as nx
from networkx.drawing.nx_pydot import to_pydot

# https://dreampuf.github.io/GraphvizOnline  에서 볼 수 있음

# 그래프 구성
G = nx.Graph()
G.add_node("O", label="O")
G.add_node("H1", label="H")
G.add_node("H2", label="H")
G.add_edge("H1", "O")
G.add_edge("H2", "O")

# DOT 생성
dot = to_pydot(G)
# dot.write_raw("lewis.dot")
print(dot.to_string())
