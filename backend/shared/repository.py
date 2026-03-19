from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Generic, TypeVar


T = TypeVar("T")


class InMemoryRepository(Generic[T]):
    def __init__(self) -> None:
        self._items: dict[str, T] = {}

    def add(self, item: T) -> T:
        self._items[self._get_id(item)] = item
        return item

    def get(self, item_id: str) -> T | None:
        return self._items.get(item_id)

    def update(self, item: T) -> T:
        item_id = self._get_id(item)
        if item_id not in self._items:
            raise KeyError(item_id)
        self._items[item_id] = item
        return item

    def delete(self, item_id: str) -> None:
        self._items.pop(item_id, None)

    def all(self) -> list[T]:
        return list(self._items.values())

    def find(self, predicate) -> list[T]:
        return [item for item in self._items.values() if predicate(item)]

    @staticmethod
    def _get_id(item: T) -> str:
        item_id = getattr(item, "id", None)
        if not item_id:
            raise ValueError("repository items must have an id")
        return str(item_id)


def clone_and_update(item: T, **changes) -> T:
    return replace(item, **changes)

