#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

__author__ = "Christian Heider Nielsen"

__all__ = ["SumTree"]

from typing import Any, Iterator

from draugr import add_indent


class SumTree:
    """

"""

    def __init__(
        self, capacity: int, round_capacity_to_nearest_power_of_two: bool = True
    ):
        """

@param capacity:
"""

        if round_capacity_to_nearest_power_of_two:
            capacity = 2 ** round(math.log(capacity, 2))

        self.capacity = capacity
        self.num_tree_levels = math.ceil(math.log(capacity, 2)) + 1
        self._tree_size = 2 ** self.num_tree_levels - 1
        self._trunc_tree_size = 2 * self.capacity - 1
        self._tree = [0.0 for _ in range(self._tree_size)]
        self._data = [None for _ in range(self.capacity)]
        self._num_entries = 0
        self._cursor = 0

    def _propagate(self, tree_index: int, delta: float) -> None:
        """
propagates changes through parents to root

@param tree_index:
@param delta:
@return:
"""

        parent_node = math.floor((tree_index - 1) / 2)
        self._tree[parent_node] += delta
        if parent_node != 0:
            self._propagate(parent_node, delta)

    def _retrieve_leaf_recursive(self, tree_index: int, sum: float) -> Any:
        """

@param sum:
@param tree_index:
@return:
"""
        left_child = 2 * tree_index + 1
        right_child = left_child + 1

        if left_child >= self._trunc_tree_size:  # If we reach bottom, end the search
            return tree_index

        left_sum = self._tree[left_child]

        if sum <= left_sum:
            return self._retrieve_leaf_recursive(left_child, sum)
        else:
            return self._retrieve_leaf_recursive(right_child, sum - left_sum)

    def _retrieve_leaf(self, sum: float) -> Any:
        """

@param sum:
@return:
"""
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if (
                left_child_index >= self._trunc_tree_size
            ):  # If we reach bottom, end the search
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                left_sum = self._tree[left_child_index]
                if sum <= left_sum:
                    parent_index = left_child_index
                else:
                    sum -= left_sum
                    parent_index = right_child_index

        return leaf_index

    def get(self, sum: float, *, normalised_sum: bool = True) -> tuple:
        """
(leaf_index, data_index, self._tree[leaf_index], self._data[data_index])
@param sum:
@param normalised_sum:
@return:
"""
        if normalised_sum:
            sum *= self.total

        leaf_index = self._retrieve_leaf(sum)
        data_index = leaf_index - self.capacity + 1
        return (leaf_index, data_index, self._tree[leaf_index], self._data[data_index])

    def push(self, data: Any, sum: float) -> None:
        """

@param data:
@param sum:
@return:
"""

        self._data[self._cursor] = data
        self.update_leaf(self._cursor + self.capacity - 1, sum)
        self._cursor = (self._cursor + 1) % self.capacity
        self._num_entries = min(self._num_entries + 1, self.capacity)

    def update_leaf(self, leaf_index: int, new_sum: float) -> None:
        """

@param leaf_index:
@param new_sum:
@return:
"""
        change = new_sum - self._tree[leaf_index]
        self._tree[leaf_index] = new_sum

        if self.num_tree_levels > 1:
            self._propagate(leaf_index, change)

    @property
    def total(self) -> float:
        """
Sum of all leaf nodes

@return:
"""
        return self._tree[0]

    def print_tree(self) -> None:
        """
WARNING HEAVY with big trees!
@return:
"""
        print("-" * 9)
        for k in range(self.num_tree_levels):
            start = 2 ** k - 1
            end = 2 ** (k + 1) - 1
            print(
                add_indent(
                    "\t".join([f"{self._tree[j]:.2f}" for j in range(start, end)]),
                    self.num_tree_levels - k,
                )
            )
        print("-" * 9)
        print("\t".join([f"{j:.2f}" for j in self._data]))
        print("-" * 9)

    def __len__(self) -> int:
        return self._num_entries

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __call__(self, sum: float = 1.0, *, normalised_sum: bool = True) -> Any:
        return self.get(sum, normalised_sum=normalised_sum)


if __name__ == "__main__":

    def stest_experience_buffer():
        capacity = int(2 ** 16)
        s = SumTree(capacity)
        for i in range(capacity):
            s.push(i, 1 / (i + 1))
        for i in range(capacity):
            s.push(i, i)
        print(s.get(0))
        print(s.get(0.5))
        print(s.get(1.0))
        print(len(s))
        print(next(iter(s)))

    stest_experience_buffer()
