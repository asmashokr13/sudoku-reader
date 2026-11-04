"""Textbook Romania road map (Russell & Norvig): optimal Arad->Bucharest cost is 418."""
EDGES = [("Arad", "Zerind", 75), ("Arad", "Sibiu", 140), ("Arad", "Timisoara", 118), ("Zerind", "Oradea", 71),
         ("Oradea", "Sibiu", 151), ("Timisoara", "Lugoj", 111), ("Lugoj", "Mehadia", 70),
         ("Mehadia", "Drobeta", 75), ("Drobeta", "Craiova", 120), ("Craiova", "Rimnicu Vilcea", 146),
         ("Craiova", "Pitesti", 138), ("Sibiu", "Fagaras", 99), ("Sibiu", "Rimnicu Vilcea", 80),
         ("Rimnicu Vilcea", "Pitesti", 97), ("Fagaras", "Bucharest", 211), ("Pitesti", "Bucharest", 101),
         ("Bucharest", "Giurgiu", 90), ("Bucharest", "Urziceni", 85), ("Urziceni", "Hirsova", 98),
         ("Urziceni", "Vaslui", 142), ("Hirsova", "Eforie", 86), ("Vaslui", "Iasi", 92), ("Iasi", "Neamt", 87)]
SLD = {"Arad": 366, "Bucharest": 0, "Craiova": 160, "Drobeta": 242, "Eforie": 161, "Fagaras": 176, "Giurgiu": 77,
       "Hirsova": 151, "Iasi": 226, "Lugoj": 244, "Mehadia": 241, "Neamt": 234, "Oradea": 380, "Pitesti": 100,
       "Rimnicu Vilcea": 193, "Sibiu": 253, "Timisoara": 329, "Urziceni": 80, "Vaslui": 199, "Zerind": 374}
ROMANIA_UCS_COST = 418            # Arad-Sibiu-Rimnicu Vilcea-Pitesti-Bucharest
ROMANIA_GREEDY_COST = 450         # Arad-Sibiu-Fagaras-Bucharest


class RomaniaProblem:
    def __init__(self):
        self.adj = {}
        for a, b, w in EDGES:
            self.adj.setdefault(a, []).append((b, w)); self.adj.setdefault(b, []).append((a, w))

    def initial_state(self): return "Arad"
    def is_goal(self, s): return s == "Bucharest"
    def heuristic(self, s): return float(SLD[s])
    def successors(self, s): return [(n, n, w) for n, w in self.adj[s]]
