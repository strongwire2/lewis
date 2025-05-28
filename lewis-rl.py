import gymnasium as gym
import numpy as np
import networkx as nx
from gymnasium import spaces

# 간단한 원자 정보: 기호, 원자가 전자 수
ATOM_INFO = {
    'H': {'valence': 1},
    'O': {'valence': 2},
    'C': {'valence': 4},
    # 필요시 추가
}


class LewisEnv(gym.Env):
    def __init__(self, atoms=['O', 'H', 'H']):
        self.atoms = atoms
        self.num_atoms = len(atoms)
        self.atom_data = [dict(symbol=s, valence=ATOM_INFO[s]['valence'], bonds=0) for s in atoms]

        self.graph = nx.Graph()
        self.remaining = list(range(self.num_atoms))
        self.placed = []

        # Observation: adjacency matrix + atom valence status
        self.observation_space = spaces.Dict({
            'adj': spaces.Box(low=0, high=1, shape=(self.num_atoms, self.num_atoms), dtype=np.int32),
            'valence': spaces.Box(low=0, high=4, shape=(self.num_atoms,), dtype=np.int32),
            'mask': spaces.MultiBinary(self.num_atoms)
        })

        # Action: 선택할 원자 인덱스, 붙일 대상 인덱스
        self.action_space = spaces.MultiDiscrete([self.num_atoms, self.num_atoms])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.graph.clear()
        self.remaining = list(range(self.num_atoms))
        self.placed = []
        self.atom_data = [dict(symbol=s, valence=ATOM_INFO[s]['valence'], bonds=0) for s in self.atoms]

        # 시작: 첫 원자 선택 (랜덤)
        first = self.np_random.choice(self.remaining)
        self.graph.add_node(first)
        self.remaining.remove(first)
        self.placed.append(first)

        return self._get_obs(), {}

    def step(self, action):
        a_idx, b_idx = action
        done = False
        reward = 0.0

        if a_idx not in self.remaining or b_idx not in self.placed:
            reward = -1.0  # invalid action
            done = True
            return self._get_obs(), reward, done, False, {}

        a = self.atom_data[a_idx]
        b = self.atom_data[b_idx]

        if a['bonds'] >= a['valence'] or b['bonds'] >= b['valence']:
            reward = -1.0
            done = True
            return self._get_obs(), reward, done, False, {}

        self.graph.add_node(a_idx)
        self.graph.add_edge(a_idx, b_idx)
        a['bonds'] += 1
        b['bonds'] += 1
        self.remaining.remove(a_idx)
        self.placed.append(a_idx)

        if len(self.remaining) == 0:
            done = True
            reward = self._evaluate()

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        adj = nx.to_numpy_array(self.graph, nodelist=list(range(self.num_atoms)), dtype=int)
        valence = np.array([a['valence'] - a['bonds'] for a in self.atom_data])
        mask = np.zeros(self.num_atoms, dtype=int)
        mask[self.remaining] = 1
        return {
            'adj': adj,
            'valence': valence,
            'mask': mask
        }

    def _evaluate(self):
        # 간단한 평가: 모든 원자가 valence 만족하면 1.0, 아니면 페널티
        score = 0
        for atom in self.atom_data:
            if atom['bonds'] == atom['valence']:
                score += 1
            else:
                score -= 1
        return score / self.num_atoms
