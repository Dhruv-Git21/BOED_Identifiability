"""Tests for graph utilities."""

import numpy as np
import pytest
from causal_boed.graphs.dag import DAG, sample_random_dag, dag_to_cpdag, find_ambiguous_edges


class TestDAG:
    """Tests for DAG class."""
    
    def test_dag_initialization(self):
        """Test DAG initialization."""
        adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        dag = DAG(adj)
        assert dag.n_nodes == 3
        assert dag.edge_count() == 2
    
    def test_dag_acyclicity_check(self):
        """Test DAG detects cycles."""
        # Cyclic adjacency
        adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        with pytest.raises(ValueError, match="cycles"):
            DAG(adj)
    
    def test_dag_parent_child_queries(self):
        """Test parent and child queries."""
        adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        dag = DAG(adj)
        
        assert dag.get_parents(0) == []
        assert dag.get_parents(1) == [0]
        assert dag.get_children(0) == [1]
        assert dag.get_children(1) == [2]
    
    def test_dag_topological_sort(self):
        """Test topological sorting."""
        adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        dag = DAG(adj)
        order = dag.topological_sort()
        
        # 0 should come before 1, 1 before 2
        assert order.index(0) < order.index(1)
        assert order.index(1) < order.index(2)
    
    def test_dag_equality(self):
        """Test DAG equality."""
        adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        dag1 = DAG(adj)
        dag2 = DAG(adj.copy())
        
        assert dag1 == dag2
    
    def test_dag_copy(self):
        """Test DAG copy."""
        adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        dag1 = DAG(adj)
        dag2 = dag1.copy()
        
        assert dag1 == dag2
        assert dag1 is not dag2


class TestRandomDAGGeneration:
    """Tests for random DAG generation."""
    
    def test_random_dag_generation(self):
        """Test random DAG generation."""
        dag = sample_random_dag(num_nodes=5, expected_degree=1.5, seed=42)
        
        assert dag.n_nodes == 5
        assert dag.is_acyclic()
    
    def test_random_dag_expected_degree(self):
        """Test expected degree is roughly correct."""
        dags = [
            sample_random_dag(num_nodes=10, expected_degree=1.0, seed=i)
            for i in range(10)
        ]
        
        degrees = [dag.edge_count() for dag in dags]
        mean_degree = np.mean(degrees)
        
        # Should be roughly 10 (expected edges)
        assert 5 < mean_degree < 15
    
    def test_random_dag_reproducibility(self):
        """Test reproducible random DAG generation."""
        dag1 = sample_random_dag(5, 1.5, seed=42)
        dag2 = sample_random_dag(5, 1.5, seed=42)
        
        assert dag1 == dag2


class TestCPDAGConversion:
    """Tests for CPDAG conversion."""
    
    def test_cpdag_from_dag(self):
        """Test CPDAG conversion."""
        adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        dag = DAG(adj)
        skeleton, oriented = dag_to_cpdag(dag)
        
        # Skeleton should have undirected edges
        assert skeleton[0, 1] == 1
        assert skeleton[1, 0] == 1
        assert skeleton[1, 2] == 1
        assert skeleton[2, 1] == 1
        
        # Oriented should have directed edges
        assert (0, 1) in oriented
        assert (1, 2) in oriented
        assert (1, 0) not in oriented


class TestAmbiguousEdges:
    """Tests for ambiguous edge detection."""
    
    def test_find_ambiguous_edges(self):
        """Test finding ambiguous edges."""
        # CPDAG with some undirected edges
        cpdag = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        
        ambiguous = find_ambiguous_edges(cpdag)
        
        # All edges are ambiguous (both directions)
        assert len(ambiguous) == 3
