from typing import Any, Callable, Self

import numpy as np


class Node:
    def __init__(
        self: Self,
        name: str,
        value: Callable | float | int,
        arity: int,
        repr_func: Callable | None = None,
    ):
        self.name = name
        self.value = value
        self.arity = arity
        self.repr_func = repr_func

    def __str__(self, level=0):
        ret = "\t" * level + self.name + "\n"
        # ret = self.name

        return ret


class ExpressionTree:
    def __init__(
        self: Self,
        root_name: str,
        root_value: Any,
        root_arity: int,
        repr_func: Callable | None = None,
    ):
        self.nodes = [Node(root_name, root_value, root_arity, repr_func)]

    @classmethod
    def init_from_list(cls: type[Self], nodes: list[Node]) -> Self:
        tree = cls.__new__(cls)
        tree.nodes = nodes
        return tree

    def __str__(self: Self):
        def _str(node_index: int, level: int) -> str:
            node = self.nodes[node_index]
            ret = node.__str__(level)
            child_index = node_index + 1
            for _ in range(node.arity):
                ret += _str(child_index, level + 1)
                child_index += self._subtree_size(child_index)
            return ret

        return _str(0, 0)

    def __len__(self: Self) -> int:
        return len(self.nodes)

    def __getitem__(self: Self, index: int) -> Node:
        return self.nodes[index]

    def _subtree_size(self: Self, node_index: int) -> int:
        node = self.nodes[node_index]
        size = 1
        child_index = node_index + 1
        for _ in range(node.arity):
            child_size = self._subtree_size(child_index)
            size += child_size
            child_index += child_size
        return size

    def add_node(
        self: Self,
        name: str,
        value: Callable | float | int,
        arity: int,
        repr_func: Callable | None = None,
    ):
        self.nodes.append(Node(name, value, arity, repr_func))

    def replace_node_with_child(
        self: Self,
        node_index: int,
        replaced_by_child: int,
    ):
        # Calculate the size of the entire subtree being replaced.
        size_to_replace = self._subtree_size(node_index)

        # Find the starting index of the child subtree that will replace the parent.
        child_start_index = node_index + 1
        for _ in range(replaced_by_child):
            child_start_index += self._subtree_size(child_start_index)

        # Get the full subtree of the child.
        child_subtree_size = self._subtree_size(child_start_index)
        child_subtree = self.nodes[
            child_start_index : child_start_index + child_subtree_size
        ]

        # Replace the original subtree slice with the child's full subtree.
        self.nodes[node_index : node_index + size_to_replace] = child_subtree

    def remove_subtree(self: Self, node_index: int):
        size = self._subtree_size(node_index)
        del self.nodes[node_index : node_index + size]

    def replace_subtree(
        self: Self,
        node_index: int,
        name: str,
        value: Any,
        arity: int,
        repr_func: Callable | None = None,
    ):
        size = self._subtree_size(node_index)
        self.nodes[node_index : node_index + size] = [
            Node(name, value, arity, repr_func)
        ]

    def evaluate(self: Self, X: np.ndarray, index: int = 0) -> np.ndarray:
        """
        Evaluates the expression tree at a given node index.

        args:
            X: np.ndarray
                Input data of shape (n_samples, n_features).
            index: int
                Index of the node to evaluate from. Defaults to 0 (the root node).
        """

        def _eval(node_index: int) -> tuple[np.ndarray, int]:
            """Recursively evaluates a subtree and returns the result and its size."""
            node = self.nodes[node_index]
            size = 1
            if node.arity == 0:
                if callable(node.value):
                    val = node.value(X)
                else:
                    val = np.full(X.shape[0], node.value)
                return val, size
            else:
                child_values = []
                child_index = node_index + 1
                for _ in range(node.arity):
                    child_val, child_size = _eval(child_index)
                    child_values.append(child_val)
                    child_index += child_size
                    size += child_size
                with np.errstate(invalid="ignore", divide="ignore"):
                    return node.value(*child_values), size

        val, _ = _eval(index)
        return val

    def _is_subtree_equal(self: Self, index1: int, index2: int) -> bool:
        """Checks if two subtrees are equal."""
        node1 = self.nodes[index1]
        node2 = self.nodes[index2]

        if node1.name != node2.name or node1.arity != node2.arity:
            return False

        size1 = self._subtree_size(index1)
        size2 = self._subtree_size(index2)

        if size1 != size2:
            return False

        for offset in range(size1):
            n1 = self.nodes[index1 + offset]
            n2 = self.nodes[index2 + offset]
            if n1.name != n2.name or n1.arity != n2.arity:
                return False

        return True

    def copy(self: Self) -> Self:
        """Create a deep copy of the tree for comparison in tests."""
        copied_nodes = [Node(n.name, n.value, n.arity, n.repr_func) for n in self.nodes]
        return ExpressionTree.init_from_list(copied_nodes)


def simplify_tree(tree: ExpressionTree, X: np.ndarray) -> None:
    """
    Simplifies the expression tree while preserving level curves (zero sets up to
    constant offset). This is done by removing operations that don't change the
    topology of the zero set.

    Position-aware simplification ensures that certain rules only apply at the
    root level to preserve level set topology correctly:

    - Constants (+c, *c) only removed at root/output edge (depth == 0)
    - Monotonic ops (log, sqrt, exp) only removed at root level (depth == 0)
    - Identity rules (x-x, x/x, *1, /1, +0, -0) work at any depth

    Works recursively with depth tracking to ensure correct simplification.
    """

    def _simplify_subtree(index: int, depth: int) -> tuple[bool, int]:
        """
        Recursively simplifies a subtree starting at the given index.

        Args:
            index: The node index to start simplification from
            depth: Depth from root (0 for root, increments for nested nodes)

        Returns:
            tuple[bool, int]: (changed, next_index)
                - changed: True if a simplification was made (requiring restart)
                - next_index: The index of the next sibling node
        """
        if index >= len(tree.nodes):
            return False, index

        node = tree.nodes[index]

        # Try simplification at current node with depth check
        if node.arity == 1:
            # Unary operations: remove if they're monotonic and preserve zeros
            # ONLY at root level (depth == 0) to preserve level set topology
            if depth == 0 and node.name in {"log", "sqrt", "exp"}:
                tree.replace_node_with_child(index, 0)
                return True, 0  # Signal restart from root

        elif node.arity == 2:
            if node.name not in {"add", "sub", "mul", "div"}:
                pass  # Continue to children
            else:
                left_index = index + 1
                right_index = index + 1 + tree._subtree_size(index + 1)

                if right_index < len(tree.nodes):
                    left = tree.nodes[left_index]
                    right = tree.nodes[right_index]

                    # Addition/subtraction with constant: doesn't change zero set (just shifts)
                    # ONLY at root level (depth == 0) to preserve level set topology
                    if depth == 0 and node.name in {"add", "sub"}:
                        if left.arity == 0 and isinstance(left.value, (int, float)):
                            tree.replace_node_with_child(index, 1)
                            return True, 0  # Signal restart from root
                        elif right.arity == 0 and isinstance(right.value, (int, float)):
                            tree.replace_node_with_child(index, 0)
                            return True, 0  # Signal restart from root

                    # Multiplication/division by constant: doesn't change zero set
                    # ONLY at root level (depth == 0) to preserve level set topology
                    elif depth == 0 and node.name in {"mul", "div"}:
                        if left.arity == 0 and isinstance(left.value, (int, float)):
                            if left.value != 0:  # Avoid div by zero or mul by zero
                                tree.replace_node_with_child(index, 1)
                                return True, 0  # Signal restart from root
                        elif right.arity == 0 and isinstance(right.value, (int, float)):
                            if right.value != 0:
                                tree.replace_node_with_child(index, 0)
                                return True, 0  # Signal restart from root

                    # Identical subtrees: identity rules work at any depth
                    if tree._is_subtree_equal(left_index, right_index):
                        if node.name == "sub":  # x - x = 0
                            tree.replace_subtree(index, "constant", 0.0, 0)
                            return True, 0  # Signal restart from root
                        elif node.name == "div":  # x / x = 1
                            tree.replace_subtree(index, "constant", 1.0, 0)
                            return True, 0  # Signal restart from root
                        elif node.name == "mul":  # x * x doesn't change zeros if x >= 0
                            result = tree.evaluate(X, left_index)
                            if (result >= 0).all():
                                tree.replace_node_with_child(index, 0)
                                return True, 0  # Signal restart from root

        # Recurse into children with incremented depth
        child_index = index + 1
        for _ in range(node.arity):
            changed, next_idx = _simplify_subtree(child_index, depth + 1)
            if changed:
                return True, 0  # Signal restart from root
            child_index = next_idx

        return False, index + tree._subtree_size(index)

    # Traverse recursively, restart whenever a simplification is made
    while True:
        changed, _ = _simplify_subtree(0, 0)
        if not changed:
            break
