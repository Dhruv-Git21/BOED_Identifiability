"""DAG (Directed Acyclic Graph) utilities and random DAG generation."""

import numpy as np
import networkx as nx
from typing import Tuple, Set, List, Optional
from causal_boed.utils.rng import get_rng


class DAG:
    """
    Directed Acyclic Graph representation for causal models.
    
    Internally uses adjacency matrix: B[i,j] = 1 if i -> j.
    """
    
    def __init__(self, adjacency: np.ndarray, labels: Optional[List[str]] = None):
        """
        Initialize DAG.
        
        Args:
            adjacency: (n, n) binary adjacency matrix where B[i,j]=1 means i->j
            labels: Optional variable names
        """
        self.adjacency = adjacency.astype(bool).astype(int)
        self.n_nodes = adjacency.shape[0]
        self.labels = labels or [f"X{i}" for i in range(self.n_nodes)]
        
        # Validate acyclicity
        if not self.is_acyclic():
            raise ValueError("Adjacency matrix contains cycles")
    
    def is_acyclic(self) -> bool:
        """Check if graph is acyclic (DAG)."""
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n_nodes))
        edges = np.argwhere(self.adjacency > 0)
        G.add_edges_from(edges)
        return nx.is_directed_acyclic_graph(G)
    
    def get_parents(self, node: int) -> List[int]:
        """Get parent nodes of given node."""
        return list(np.where(self.adjacency[:, node] > 0)[0])
    
    def get_children(self, node: int) -> List[int]:
        """Get children nodes of given node."""
        return list(np.where(self.adjacency[node, :] > 0)[0])
    
    def topological_sort(self) -> List[int]:
        """Return topological ordering of nodes."""
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n_nodes))
        edges = np.argwhere(self.adjacency > 0)
        G.add_edges_from(edges)
        return list(nx.topological_sort(G))
    
    def copy(self) -> "DAG":
        """Return a copy of this DAG."""
        return DAG(self.adjacency.copy(), self.labels.copy())
    
    def edge_count(self) -> int:
        """Number of edges."""
        return int(np.sum(self.adjacency))
    
    def __repr__(self) -> str:
        return f"DAG(nodes={self.n_nodes}, edges={self.edge_count()})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DAG):
            return False
        return np.array_equal(self.adjacency, other.adjacency)
    
    def __hash__(self) -> int:
        """Hash based on adjacency matrix (as tuple)."""
        return hash(tuple(self.adjacency.flat))


def sample_random_dag(
    num_nodes: int,
    expected_degree: float = 1.5,
    seed: Optional[int] = None
) -> DAG:
    """
    Generate a random DAG using topological ordering + edge inclusion.
    
    Args:
        num_nodes: Number of nodes
        expected_degree: Expected number of edges per node (controls sparsity)
        seed: Random seed
        
    Returns:
        Random DAG
    """
    rng = get_rng() if seed is None else np.random.default_rng(seed)
    
    # Edge probability to achieve expected degree
    edge_prob = min(expected_degree / num_nodes, 1.0)
    
    # Topological ordering (fixed permutation)
    perm = rng.permutation(num_nodes)
    
    # Build adjacency matrix respecting topological order
    adjacency = np.zeros((num_nodes, num_nodes), dtype=int)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Only add edge from earlier to later in topological order
            if rng.uniform() < edge_prob:
                u, v = perm[i], perm[j]
                adjacency[u, v] = 1
    
    return DAG(adjacency)


def dag_to_cpdag(dag: DAG) -> Tuple[np.ndarray, Set[Tuple[int, int]]]:
    """
    Convert DAG to Completed Partially Directed Acyclic Graph (CPDAG).
    
    Returns the skeleton (undirected edges) and the set of oriented edges.
    This is a simplified version; for full CPDAG, use PC algorithm.
    
    Args:
        dag: Input DAG
        
    Returns:
        (skeleton, oriented_edges) where skeleton[i,j] = skeleton[j,i] = 1 if edge,
        and oriented_edges is set of (i,j) meaning i->j (or j->i for undirected)
    """
    skeleton = (dag.adjacency + dag.adjacency.T) > 0
    oriented = set()
    for i, j in zip(*np.where(dag.adjacency)):
        oriented.add((i, j))
    
    return skeleton.astype(int), oriented


def find_ambiguous_edges(cpdag: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find ambiguous (undirected) edges in a CPDAG.
    
    Args:
        cpdag: Completed PDAG adjacency (might be partially directed)
        
    Returns:
        List of (i, j) where edge is ambiguous (not clearly directed)
    """
    ambiguous = []
    n = cpdag.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if cpdag[i, j] > 0 and cpdag[j, i] > 0:
                # Undirected edge (both directions present)
                ambiguous.append((i, j))
    return ambiguous


def find_markov_equivalence_class(dag: DAG) -> List[DAG]:
    """
    Compute the Markov equivalence class of a DAG.
    For small graphs, enumerate all DAGs with same conditional independencies.
    
    This is a simplified version; for large graphs, use CPDAG theory.
    
    Args:
        dag: Input DAG
        
    Returns:
        List of DAGs in the same Markov equivalence class
    """
    # For now, return the DAG itself and any reversible edges
    # A proper implementation would use CPDAG and generate all DAGs from it
    mec = [dag]
    
    # Find edges that could be reversible (no v-structures involved)
    n = dag.n_nodes
    for i in range(n):
        for j in range(i + 1, n):
            if dag.adjacency[i, j]:
                # Try reversing edge i->j to j->i
                adj_rev = dag.adjacency.copy()
                adj_rev[i, j] = 0
                adj_rev[j, i] = 1
                try:
                    rev_dag = DAG(adj_rev)
                    # Check if different and not already in MEC
                    if rev_dag not in mec:
                        mec.append(rev_dag)
                except ValueError:
                    # Would create cycle
                    pass
    
    return mec
