"""Generic best-first search with counters: UCS, Greedy best-first and A*.

A problem object provides:
    initial_state(), is_goal(state), successors(state) -> iterable of (action, next_state, cost),
    heuristic(state) -> float  (float('inf') = dead end, only used by greedy / astar)
The goal test is applied when a node is *expanded* (popped), which keeps UCS and A* optimal.
"""
import heapq, itertools, time
from dataclasses import dataclass, field

INF = float("inf")


@dataclass
class SearchResult:
    found: bool
    state: object = None
    cost: float = INF
    path: list = field(default_factory=list)
    generated: int = 0
    expanded: int = 0
    pruned: int = 0
    goal_tests: int = 0
    max_frontier: int = 0
    seconds: float = 0.0
    aborted: bool = False
    trace: list = field(default_factory=list)      # (expansion#, g, h, state) for the first trace_limit nodes

    def counters(self):
        return dict(generated=self.generated, expanded=self.expanded, pruned=self.pruned,
                    goal_tests=self.goal_tests, max_frontier=self.max_frontier)


def best_first_search(problem, strategy="ucs", max_expansions=None, trace_limit=0) -> SearchResult:
    if strategy not in ("ucs", "greedy", "astar"):
        raise ValueError(strategy)
    t0 = time.perf_counter()
    tie = itertools.count()
    res = SearchResult(False)

    def prio(g, h):
        if strategy == "ucs":
            return (g, 0.0)
        if strategy == "greedy":
            return (h, g)
        return (g + h, h)

    s0 = problem.initial_state()
    h0 = problem.heuristic(s0) if strategy != "ucs" else 0.0
    best_g = {s0: 0.0}
    parent = {s0: (None, None)}
    frontier = [(*prio(0.0, h0), next(tie), s0, 0.0, h0)]
    res.generated = 1
    while frontier:
        res.max_frontier = max(res.max_frontier, len(frontier))
        *_, state, g, h = heapq.heappop(frontier)
        if g > best_g.get(state, INF):
            continue                                      # stale entry
        res.goal_tests += 1
        if problem.is_goal(state):
            res.found, res.state, res.cost = True, state, g
            path, s = [], state
            while parent[s][0] is not None:
                s, a = parent[s][0], parent[s][1]
                path.append(a)
            res.path = path[::-1]
            break
        if max_expansions is not None and res.expanded >= max_expansions:
            res.aborted = True
            break
        res.expanded += 1
        if len(res.trace) < trace_limit:
            res.trace.append((res.expanded, g, h, state))
        for action, nxt, cost in problem.successors(state):
            ng = g + cost
            if ng >= best_g.get(nxt, INF):
                continue
            nh = problem.heuristic(nxt) if strategy != "ucs" else 0.0
            if nh == INF:
                res.pruned += 1
                continue
            best_g[nxt] = ng
            parent[nxt] = (state, action)
            heapq.heappush(frontier, (*prio(ng, nh), next(tie), nxt, ng, nh))
            res.generated += 1
    res.seconds = time.perf_counter() - t0
    return res
