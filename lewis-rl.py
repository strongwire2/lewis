import networkx as nx
import random
import numpy as np

class LewisCO2Env:
    def __init__(self):
        self.reset()

    def reset(self):
        self.graph = nx.Graph()
        self.graph.add_node("C_0", label="C", lone_e=4)
        self.graph.add_node("O_0", label="O", lone_e=6)
        self.graph.add_node("O_1", label="O", lone_e=6)
        self.graph.add_edge("C_0", "O_0", bond=1)
        self.graph.add_edge("C_0", "O_1", bond=1)
        self.total_valence_e = 16
        return self._get_state()

    def _get_state(self):
        bonds = tuple((u, v, d["bond"]) for u, v, d in self.graph.edges(data=True))
        lone_e = tuple((n, self.graph.nodes[n]["lone_e"]) for n in self.graph.nodes)
        return (bonds, lone_e)

    def _electron_count(self):
        shared_e = sum(d['bond'] * 2 for _, _, d in self.graph.edges(data=True))
        lone_e = sum(self.graph.nodes[n]['lone_e'] for n in self.graph.nodes)
        return shared_e + lone_e

    def _octet_penalty(self):
        penalty = 0
        for n in self.graph.nodes:
            shared = sum(self.graph[u][v]['bond'] * 2 for u, v in self.graph.edges(n))
            lone = self.graph.nodes[n]['lone_e']
            total = shared + lone
            if self.graph.nodes[n]['label'] == 'H':
                penalty += abs(2 - total)
            else:
                penalty += abs(8 - total)
        return penalty

    def step(self, action):
        # action: (u, v, +1|-1) -> modify bond
        u, v, delta = action
        if not self.graph.has_edge(u, v):
            return self._get_state(), -1, False, {}

        current_bond = self.graph[u][v]['bond']
        new_bond = current_bond + delta

        if not 0 < new_bond <= 3:
            return self._get_state(), -1, False, {}

        # adjust lone pairs accordingly
        diff_e = delta * 2
        if self.graph.nodes[u]['lone_e'] >= delta and self.graph.nodes[v]['lone_e'] >= delta:
            self.graph[u][v]['bond'] = new_bond
            self.graph.nodes[u]['lone_e'] -= delta
            self.graph.nodes[v]['lone_e'] -= delta
        else:
            return self._get_state(), -1, False, {}

        total_e = self._electron_count()
        done = (total_e == self.total_valence_e and self._octet_penalty() == 0)
        reward = -abs(self.total_valence_e - total_e) - self._octet_penalty()
        return self._get_state(), reward, done, {}

    def render(self):
        for u, v, d in self.graph.edges(data=True):
            print(f"{u} -- {v}, bond={d['bond']}")
        for n in self.graph.nodes:
            print(f"{n}: lone_e={self.graph.nodes[n]['lone_e']}")
