import pytest
import numpy as np

from primel.tree import ExpressionTree, Node, simplify_tree


@pytest.fixture
def pos_data() -> np.ndarray:
    """Positive only data"""
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])


@pytest.fixture
def pos_neg_data() -> np.ndarray:
    """Positive and Negative data"""
    return np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])


@pytest.fixture
def periodic_data() -> np.ndarray:
    """Data within between (0, 2pi) not including endpoints."""
    return np.linspace(0.1, 2 * np.pi - 0.1, 10)


class TestExpressionTree:
    """Tests for the ExpressionTree class methods."""

    def test_tree_creation_and_str(self):
        """Test basic tree creation and its string representation."""
        tree = ExpressionTree("add", lambda a, b: a + b, 2)
        tree.add_node("x0", lambda x: x, 0)
        tree.add_node("const", 2.0, 0)

        assert str(tree).startswith("add")
        assert "x0" in str(tree)
        assert "const" in str(tree)
        assert len(tree.nodes) == 3

    def test_subtree_size(self):
        """Test the _subtree_size method with nested children."""
        # Represents mul(add(x0, 1), x1)
        nodes_list = [
            Node("mul", np.multiply, 2),
            Node("add", np.add, 2),
            Node("x0", lambda x: x[:, 0], 0),
            Node("const", lambda x: 1, 0),
            Node("x1", lambda x: x[:, 1], 0),
        ]
        tree = ExpressionTree.init_from_list(nodes_list)

        assert tree._subtree_size(0) == 5  # whole tree mul(...)
        assert tree._subtree_size(1) == 3  # subtree add(...)
        assert tree._subtree_size(2) == 1  # leaf x0
        assert tree._subtree_size(4) == 1  # leaf x1

    def test_evaluate(self, pos_data):
        """Test the evaluate method on a simple expression."""
        # Represents (x + 5)
        tree = ExpressionTree("add", np.add, 2)
        tree.add_node("x", lambda x: x, 0)
        tree.add_node("const", 5.0, 0)

        result = tree.evaluate(pos_data)
        expected = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        np.testing.assert_array_equal(result, expected)

    def test_evaluate_nested(self, pos_data):
        """Test the evaluate method on a nested expression."""
        # Represents add(mul(x, 2), sub(x, 3))
        tree = ExpressionTree("add", np.add, 2)
        tree.add_node("mul", np.multiply, 2)
        tree.add_node("x", lambda x: x, 0)
        tree.add_node("const2", 2.0, 0)
        tree.add_node("sub", np.subtract, 2)
        tree.add_node("x", lambda x: x, 0)
        tree.add_node("const3", 3.0, 0)

        result = tree.evaluate(pos_data)
        expected = (pos_data * 2) + (pos_data - 3)
        np.testing.assert_array_equal(result, expected)

    def test_replace_subtree(self, pos_data):
        """Test replacing a subtree with a new node."""
        # Start with add(mul(x, 2), 5)
        tree = ExpressionTree("add", np.add, 2)
        tree.add_node("mul", np.multiply, 2)
        tree.add_node("x", lambda x: x, 0)
        tree.add_node("const2", 2.0, 0)
        tree.add_node("const5", 5.0, 0)

        # Replace mul(x, 2) subtree (at index 1) with just x
        tree.replace_subtree(1, "x", lambda x: x, 0)

        # Expected tree is now add(x, 5)
        assert len(tree.nodes) == 3
        assert tree.nodes[0].name == "add"
        assert tree.nodes[1].name == "x"
        assert tree.nodes[2].name == "const5"

        result = tree.evaluate(pos_data)
        expected = pos_data + 5
        np.testing.assert_array_equal(result, expected)

    def test_replace_node_with_child(self):
        # Tree is add(mul(x, 2), 5)
        tree = ExpressionTree("add", np.add, 2)
        tree.add_node("mul", np.multiply, 2)
        tree.add_node("x", lambda x: x, 0)
        tree.add_node("const2", 2.0, 0)
        tree.add_node("const5", 5.0, 0)

        tree.replace_node_with_child(0, 0)

        assert len(tree.nodes) == 3
        assert tree.nodes[0].name == "mul"


class TestSimplifyTree:
    """Tests for the heuristic simplification rules."""

    def test_simplify_log(self, pos_data):
        """Test rule: log(x) -> x"""
        tree = ExpressionTree("log", np.log, 1)
        tree.add_node("x", lambda x: x, 0)

        simplify_tree(tree, pos_data)

        assert len(tree.nodes) == 1
        assert tree.nodes[0].name == "x"

    def test_simplify_exp(self, pos_data):
        """Test rule: exp(x) -> x (matches Julia Simplify.jl line 35)"""
        tree = ExpressionTree("exp", np.exp, 1)
        tree.add_node("x", lambda x: x, 0)

        simplify_tree(tree, pos_data)

        assert len(tree.nodes) == 1
        assert tree.nodes[0].name == "x"

    def test_simplify_trig_small_range(self, pos_data, periodic_data):
        """Test rule: sin(x) -> x for small input range."""
        tree = ExpressionTree("sin", np.sin, 1)
        tree.add_node("x", lambda x: x, 0)
        # Should simplify since periodic_data is within (0, 2pi)
        simplify_tree(tree, periodic_data)

        assert len(tree.nodes) == 1
        assert tree.nodes[0].name == "x"

        tree = ExpressionTree("sin", np.sin, 1)
        tree.add_node("x", lambda x: x, 0)
        # Should not simplify since pos_data is outside (0, 2pi)
        simplify_tree(tree, pos_data)

        assert len(tree.nodes) == 2
        assert tree.nodes[0].name == "sin"

    def test_simplify_trig_large_range(self, pos_data):
        """Test rule: sin(x) should not change for large input range."""
        tree = ExpressionTree("sin", np.sin, 1)
        tree.add_node("x", lambda x: x, 0)
        X = np.linspace(0, 10, 5)  # Range is 10, which is > 2*pi

        simplify_tree(tree, X)

        assert len(tree.nodes) == 2
        assert tree.nodes[0].name == "sin"

    def test_simplify_sub_x_x(self, pos_data):
        """Test rule: sub(x, x) -> 0"""
        tree = ExpressionTree("sub", np.subtract, 2)
        tree.add_node("x", lambda x: x, 0)
        tree.add_node("x", lambda x: x, 0)

        simplify_tree(tree, pos_data)

        assert len(tree.nodes) == 1
        assert tree.nodes[0].name == "constant"
        assert tree.nodes[0].value == 0

    def test_simplify_div_x_x(self, pos_data):
        """Test rule: div(x, x) -> 1"""
        tree = ExpressionTree("div", np.divide, 2)
        tree.add_node("x", lambda x: x, 0)
        tree.add_node("x", lambda x: x, 0)

        simplify_tree(tree, pos_data)

        assert len(tree.nodes) == 1
        assert tree.nodes[0].name == "constant"
        assert tree.nodes[0].value == 1

    def test_simplify_op_with_constant_right(self, pos_data):
        """Test rule: op(x, const) -> x"""
        tree = ExpressionTree("add", np.add, 2)
        tree.add_node("x", lambda x: x, 0)
        tree.add_node("const", 5, 0)

        simplify_tree(tree, pos_data)

        assert len(tree.nodes) == 1
        assert tree.nodes[0].name == "x"

    def test_simplify_op_with_constant_left(self, pos_data):
        """Test rule: op(const, y) -> y"""
        tree = ExpressionTree("mul", np.multiply, 2)
        tree.add_node("const", 5.0, 0)
        tree.add_node("y", lambda y: y, 0)

        simplify_tree(tree, pos_data)

        assert len(tree.nodes) == 1
        assert tree.nodes[0].name == "y"

    def test_simplify_nested(self, pos_data, periodic_data):
        """Test simplification in a nested tree structure."""
        # Represents add(mul(x, 2), sub(x, 2))
        nodes_list = [
            Node("add", np.add, 2),
            Node("mul", np.multiply, 2),
            Node("x", lambda x: x, 0),
            Node("const2", 2.0, 0),
            Node("sub", np.subtract, 2),
            Node("x", lambda x: x, 0),
            Node("const2", 2.0, 0),
        ]
        tree = ExpressionTree.init_from_list(nodes_list)

        simplify_tree(tree, pos_data)

        assert len(tree.nodes) == 1
        assert tree.nodes[0].name == "x"

        # Represents sub(mul(x, y), mul(x, z))
        nodes_list = [
            Node("sub", np.subtract, 2),
            Node("mul", np.multiply, 2),
            Node("x", lambda x: x, 0),
            Node("y", lambda x: x, 0),
            Node("mul", np.multiply, 2),
            Node("x", lambda x: x, 0),
            Node("z", lambda x: x, 0),
        ]
        tree = ExpressionTree.init_from_list(nodes_list)
        simplify_tree(tree, pos_data)

        assert len(tree.nodes) == 7

        # Represents sin(cos(cos(mul(x, 0.030))))
        # With top-down approach: mul(x, 0.03) has small range, so cos simplifies it
        # Then the next cos simplifies, then sin simplifies -> all reduce to x
        nodes_list = [
            Node("sin", np.sin, 1),
            Node("cos", np.cos, 1),
            Node("cos", np.cos, 1),
            Node("mul", np.multiply, 2),
            Node("x", lambda x: x, 0),
            Node("const", 0.03, 0),
        ]
        tree = ExpressionTree.init_from_list(nodes_list)
        simplify_tree(tree, pos_data)
        assert len(tree.nodes) == 1  # All operations simplify away
        assert tree.nodes[0].name == "x"

        tree = ExpressionTree.init_from_list(nodes_list)
        simplify_tree(tree, periodic_data)
        assert len(tree.nodes) == 1
        assert tree.nodes[0].name == "x"

        # Represents mul(add(x, 3), mul(x, 3))
        # add(x, 3) -> x, mul(x, 3) -> x, then mul(x, x) -> x (since x >= 0)
        nodes_list = [
            Node("mul", np.multiply, 2),
            Node("add", np.add, 2),
            Node("x", lambda x: x, 0),
            Node("const", 3.0, 0),
            Node("mul", np.multiply, 2),
            Node("x", lambda x: x, 0),
            Node("const", 3.0, 0),
        ]
        tree = ExpressionTree.init_from_list(nodes_list)
        simplify_tree(tree, pos_data)
        assert len(tree.nodes) == 1
        assert tree.nodes[0].name == "x"

    def test_simplify_crazy(self, pos_data, periodic_data):
        """Test trees taken directly from real results."""
        # Represents sub(cos(mul(0.037, X0)), log(cos(-0.957)))
        # Right side log(cos(-0.957)) is a constant, gets removed by sub
        # Then mul(0.037, X0) is mul by constant, gets simplified to X0
        # Then cos(X0) with small range gets simplified to X0
        nodes_list = [
            Node("sub", np.subtract, 2),
            Node("cos", np.cos, 1),
            Node("mul", np.multiply, 2),
            Node("constant", 0.037, 0),
            Node("X0", lambda x: x, 0),
            Node("log", np.log, 1),
            Node("cos", np.cos, 1),
            Node("constant", -0.957, 0),
        ]
        tree = ExpressionTree.init_from_list(nodes_list)
        simplify_tree(tree, pos_data)
        assert len(tree.nodes) == 1
        assert tree.nodes[0].name == "X0"

        tree = ExpressionTree.init_from_list(nodes_list)
        simplify_tree(tree, periodic_data)
        assert len(tree.nodes) == 1
        assert tree.nodes[0].name == "X0"

        # Represents sin(tan(cos(sin(log(sqrt(cos(tan(log(div(0.030, X1))))))))))
        # div(0.03, X1) simplifies to X1, then all monotonic ops simplify away
        nodes_list = [
            Node("sin", np.sin, 1),
            Node("tan", np.tan, 1),
            Node("cos", np.cos, 1),
            Node("sin", np.sin, 1),
            Node("log", np.log, 1),
            Node("sqrt", np.sqrt, 1),
            Node("cos", np.cos, 1),
            Node("tan", np.tan, 1),
            Node("log", np.log, 1),
            Node("div", np.divide, 2),
            Node("constant", 0.03, 0),
            Node("X1", lambda x: x, 0),
        ]
        tree = ExpressionTree.init_from_list(nodes_list)
        simplify_tree(tree, pos_data)
        assert len(tree.nodes) == 1
        assert tree.nodes[0].name == "X1"

        # Represents sub(cos(tan(mul(mul(X1, 0.013), tan(X1)))), sqrt(cos(sub(log(X0), log(X0)))))
        # Right side: sub(log(X0), log(X0)) -> 0, cos(0) -> constant, sqrt -> removes, then sub with constant
        # Left side: mul(X1, 0.013) -> X1, so becomes mul(X1, tan(X1))
        nodes_list = [
            Node("sub", np.subtract, 2),
            Node("cos", np.cos, 1),
            Node("tan", np.tan, 1),
            Node("mul", np.multiply, 2),
            Node("mul", np.multiply, 2),
            Node("X1", lambda x: x, 0),
            Node("constant", 0.013, 0),
            Node("tan", np.tan, 1),
            Node("X1", lambda x: x, 0),
            Node("sqrt", np.sqrt, 1),
            Node("cos", np.cos, 1),
            Node("sub", np.subtract, 2),
            Node("log", np.log, 1),
            Node("X0", lambda x: x, 0),
            Node("log", np.log, 1),
            Node("X0", lambda x: x, 0),
        ]
        tree = ExpressionTree.init_from_list(nodes_list)
        simplify_tree(tree, pos_data)
        assert len(tree.nodes) == 4
        assert tree.nodes[0].name == "mul"

        # Represents add(mul(X1, X1), mul(X0, X0))
        # Both mul(X, X) with X >= 0 simplify to X, leaving add(X1, X0)
        nodes_list = [
            Node("add", np.add, 2),
            Node("mul", np.multiply, 2),
            Node("X1", lambda x: x, 0),
            Node("X1", lambda x: x, 0),
            Node("mul", np.multiply, 2),
            Node("X0", lambda x: x, 0),
            Node("X0", lambda x: x, 0),
        ]
        tree = ExpressionTree.init_from_list(nodes_list)
        simplify_tree(tree, pos_data)
        assert len(tree.nodes) == 3
        assert tree.nodes[0].name == "add"
