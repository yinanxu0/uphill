import random
from itertools import islice
from typing import Callable, Iterable, List, Optional

from uphill.core.utils import T
from uphill.core.audio import (
    exactly_one_not_null,
    split_sequence,
)
from uphill import loggerx


class AlgorithmMixin(Iterable):
    """
    Helper base class with methods that are supposed to work identically
    on manifest classes such as `uphill.DocumentArray`, 
    `uphill.SupervisionArray`, etc.
    """

    def filter(self, predicate: Callable[[T], bool]) -> 'T':
        """
        Return a new manifest containing only the items that satisfy ``predicate``.
        If the manifest is lazy, the filtering will also be applied lazily.

        :param predicate: a function that takes a cut as an argument and returns bool.
        :return: a filtered manifest.
        """
        cls = type(self)
        return cls.from_items(item for item in self if predicate(item))

    def map(self, transform_fn: Callable[[T], T]) -> 'T':
        """
        Apply `transform_fn` to each item in this manifest and return a new manifest.
        If the manifest is opened lazy, the transform is also applied lazily.

        :param transform_fn: A callable (function) that accepts a single item instance
            and returns a new (or the same) instance of the same type.
            E.g. with CutSet, callable accepts ``Cut`` and returns also ``Cut``.
        :return: a new ``CutSet`` with transformed cuts.
        """
        cls = type(self)
        return cls.from_items(transform_fn(item) for item in self)

    def sort(self, predicate: Callable[[T], bool]) -> 'T':
        ...
    
    def shuffle(self):
        """
        Shuffles the elements inplace.
        """
        key_value_pairs = sorted(self.data.items(), key=lambda x: random.random())
        self.data = dict(key_value_pairs)

    def split(
        self, num_splits: int, shuffle: bool = False, drop_last: bool = False
    ) -> List["T"]:
        """
        Split the `Data Array` into ``num_splits`` pieces of equal size.

        :param num_splits: Requested number of splits.
        :param shuffle: Optionally shuffle the documents order first.
        :param drop_last: determines how to handle splitting when ``len(seq)`` is not divisible
            by ``num_splits``. When ``False`` (default), the splits might have unequal lengths.
            When ``True``, it may discard the last element in some splits to ensure they are
            equally long.
        :return: A list of `Data Array` pieces.
        """
        cls = type(self)
        return [
            cls.from_items(subset)
            for subset in split_sequence(
                self, num_splits=num_splits, shuffle=shuffle, drop_last=drop_last
            )
        ]

    def subset(
        self, first: Optional[int] = None, last: Optional[int] = None
    ) -> "T":
        """
        Return a new `Data Array` according to the selected subset criterion.
        Only a single argument to ``subset`` is supported at this time.

        :param first: int, the number of first documents to keep.
        :param last: int, the number of last documents to keep.
        :return: a new `Data Array` with the subset results.
        """
        assert exactly_one_not_null(
            first, last
        ), "subset() can handle only one non-None arg."

        cls = type(self)
        if first is not None:
            assert first > 0
            if len(self) < first:
                loggerx.warning(
                    f"{self.__class__.__name__} has only {len(self)} items but first {first} requested; "
                    f"not doing anything."
                )
            slice_ids = [val for i, val in enumerate(self.ids) if i < first]
            # subset_slice = islice(self, 0, min(first, len(self)))

        if last is not None:
            assert last > 0
            if len(self) < last:
                loggerx.warning(
                    f"{self.__class__.__name__} has only {len(self)} items but last {last} requested; "
                    f"not doing anything."
                )
            slice_ids = [val for i, val in enumerate(self.ids) if i >= len(self)-last]
            # subset_slice = islice(self, max(0, len(self)-last), len(self))
        subset_slice = [self[i] for i in slice_ids]
        out = cls.from_items(subset_slice)
        return out

    def sample(self, num: int) -> List["T"]:
        assert num > 0
        random_keys = random.sample(self.ids, num)
        values = [self[random_key] for random_key in random_keys]
        return values

    def __add__(self, other) -> "T":
        assert type(self) == type(other)
        cls = type(self)
        data = self.data
        data.update(other.data)
        return cls.from_items(data.values())

