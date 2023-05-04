import math
import random
from dataclasses import dataclass
from decimal import ROUND_HALF_DOWN, ROUND_HALF_UP, Decimal
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)
from typing_extensions import Literal
import numpy as np
import torch
try:
    # Pytorch >= 1.7
    from torch.fft import irfft, rfft
except ImportError:
    from torch import irfft, rfft

from uphill.core.utils import T, Seconds


def index_by_id_and_check(manifests: Iterable[T]) -> Dict[str, T]:
    id2man = {}
    for m in manifests:
        assert m.id not in id2man, f"Duplicated manifest ID: {m.id}"
        id2man[m.id] = m
    return id2man

@dataclass
class TimeSpan:
    start: float
    end: float
    round_digits: int = 7

    def __post_init__(self):
        self.start = round(self.start, ndigits=self.round_digits)
        self.end = round(self.end, ndigits=self.round_digits)

    @property
    def duration(self):
        return round(self.end - self.start, ndigits=self.round_digits)


def overspans(spana: TimeSpan, spanb: TimeSpan) -> bool:
    """Indicates whether the left-hand-side time-span/segment covers the whole right-hand-side time-span/segment."""
    return spana.start <= spanb.start <= spanb.end <= spana.end


def perturb_num_samples(num_samples: int, factor: float) -> int:
    """Mimicks the behavior of the speed perturbation on the number of samples."""
    rounding = ROUND_HALF_UP if factor >= 1.0 else ROUND_HALF_DOWN
    return int(
        Decimal(round(num_samples / factor, ndigits=8)).quantize(0, rounding=rounding)
    )


def compute_num_samples(
    duration: Seconds, sampling_rate: int, rounding=ROUND_HALF_UP
) -> int:
    """
    Convert a time quantity to the number of samples given a specific sampling rate.
    Performs consistent rounding up or down for ``duration`` that is not a multiply of
    the sampling interval (unlike Python's built-in ``round()`` that implements banker's rounding).
    """
    return int(
        Decimal(round(duration * sampling_rate, ndigits=8)).quantize(
            0, rounding=rounding
        )
    )


def compute_num_windows(
    duration: Seconds, window_size: Seconds, frame_shift: Seconds
) -> int:
    """
    Return a number of windows obtained from signal of length equal to ``duration``
    with windows of ``window_size`` and ``frame_shift`` denoting shift between windows.
    Examples:
    ```
      (duration, window_size, frame_shift) -> num_windows # list of windows times
      (1, 6.1, 3) -> 1  # 0-1
      (3, 1, 6.1) -> 1  # 0-1
      (3, 6.1, 1) -> 1  # 0-3
      (5.9, 1, 3) -> 2  # 0-1, 3-4
      (5.9, 3, 1) -> 4  # 0-3, 1-4, 2-5, 3-5.9
      (6.1, 1, 3) -> 3  # 0-1, 3-4, 6-6.1
      (6.1, 3, 1) -> 5  # 0-3, 1-4, 2-5, 3-6, 4-6.1
      (5.9, 3, 3) -> 2  # 0-3, 3-5.9
      (6.1, 3, 3) -> 3  # 0-3, 3-6, 6-6.1
      (0.0, 3, 3) -> 0
    ```
    :param duration: Signal length in seconds.
    :param window_size: Window length in seconds
    :param frame_shift: Shift between windows in seconds.
    :return: Number of windows in signal.
    """
    # to milli-second for avoid arithmatic error
    duration_ms = duration * 1000
    window_size_ms = window_size * 1000
    frame_shift_ms = frame_shift * 1000

    n = math.ceil(max(duration_ms - window_size_ms, 0) / frame_shift_ms)
    difference_ms = duration_ms - n * frame_shift_ms - window_size_ms
    b = 0 if difference_ms < 0 else 1
    return (duration > 0) * (n + int(b))


def compute_start_duration_for_extended_cut(
    start: Seconds,
    duration: Seconds,
    new_duration: Seconds,
    direction: Literal["center", "left", "right", "random"] = "center",
) -> Tuple[Seconds, Seconds]:
    """
    Compute the new value of "start" for a time interval characterized by ``start`` and ``duration``
    that is being extended to ``new_duration`` towards ``direction``.
    :return: a new value of ``start`` and ``new_duration`` -- adjusted for possible negative start.
    """

    if new_duration <= duration:
        # New duration is shorter; do nothing.
        return start, duration

    if direction == "center":
        new_start = start - (new_duration - duration) / 2
    elif direction == "left":
        new_start = start - (new_duration - duration)
    elif direction == "right":
        new_start = start
    elif direction == "random":
        new_start = random.uniform(start - (new_duration - duration), start)
    else:
        raise ValueError(f"Unexpected direction: {direction}")

    if new_start < 0:
        # We exceeded the start of the recording.
        # We'll decrease the new_duration by the negative offset.
        new_duration = round(new_duration + new_start, ndigits=15)
        new_start = 0

    return round(new_start, ndigits=15), new_duration


def add_durations(*durations: Seconds, sampling_rate: int) -> Seconds:
    """
    Adds two durations in a way that avoids floating point precision issues.
    The durations in seconds are first converted to audio sample counts,
    then added, and finally converted back to floating point seconds.
    """
    tot_num_samples = sum(
        compute_num_samples(d, sampling_rate=sampling_rate) for d in durations
    )
    return tot_num_samples / sampling_rate


def exactly_one_not_null(*args) -> bool:
    not_null = [1 if arg is not None else 0 for arg in args]
    return sum(not_null) == 1


def split_sequence(
    seq: Sequence[Any], num_splits: int, shuffle: bool = False, drop_last: bool = False
) -> List[List[Any]]:
    """
    Split a sequence into ``num_splits`` equal parts. The element order can be randomized.
    Raises a ``ValueError`` if ``num_splits`` is larger than ``len(seq)``.

    :param seq: an input iterable (can be a manifest).
    :param num_splits: how many output splits should be created.
    :param shuffle: optionally shuffle the sequence before splitting.
    :param drop_last: determines how to handle splitting when ``len(seq)`` is not divisible
        by ``num_splits``. When ``False`` (default), the splits might have unequal lengths.
        When ``True``, it may discard the last element in some splits to ensure they are
        equally long.
    :return: a list of length ``num_splits`` containing smaller lists (the splits).
    """
    seq = list(seq)
    num_items = len(seq)
    if num_splits > num_items:
        raise ValueError(
            f"Cannot split iterable into more chunks ({num_splits}) than its number of items {num_items}"
        )
    if shuffle:
        random.shuffle(seq)
    chunk_size = num_items // num_splits

    num_shifts = num_items % num_splits
    if drop_last:
        # Equally-sized splits; discards the remainder by default, no shifts are needed
        end_shifts = [0] * num_splits
        begin_shifts = [0] * num_splits
    else:
        # Non-equally sized splits; need to shift the indices like:
        # [0, 10] -> [0, 11]    (begin_shift=0, end_shift=1)
        # [10, 20] -> [11, 22]  (begin_shift=1, end_shift=2)
        # [20, 30] -> [22, 32]  (begin_shift=2, end_shift=2)
        # for num_items=32 and num_splits=3
        end_shifts = list(range(1, num_shifts + 1)) + [num_shifts] * (
            num_splits - num_shifts
        )
        begin_shifts = [0] + end_shifts[:-1]

    split_indices = [
        [i * chunk_size + begin_shift, (i + 1) * chunk_size + end_shift]
        for i, begin_shift, end_shift in zip(
            range(num_splits), begin_shifts, end_shifts
        )
    ]
    splits = [seq[begin:end] for begin, end in split_indices]
    return splits


def ifnone(item: Optional[Any], alt_item: Any) -> Any:
    """Return ``alt_item`` if ``item is None``, otherwise ``item``."""
    return alt_item if item is None else item


class SetContainingAnything:
    def __contains__(self, item):
        return True

    def intersection(self, iterable):
        return True

def measure_overlap(lhs: Any, rhs: Any) -> float:
    """
    Given two objects with "start" and "end" attributes, return the % of their overlapped time
    with regard to the shorter of the two spans.
    ."""
    lhs, rhs = sorted([lhs, rhs], key=lambda item: item.start)
    overlapped_area = lhs.end - rhs.start
    if overlapped_area <= 0:
        return 0.0
    dur = min(lhs.end - lhs.start, rhs.end - rhs.start)
    return overlapped_area / dur


def overlaps(lhs: Any, rhs: Any) -> bool:
    """Indicates whether two time-spans/segments are overlapping or not."""
    return (
        lhs.start < rhs.end
        and rhs.start < lhs.end
        and not math.isclose(lhs.start, rhs.end)
        and not math.isclose(rhs.start, lhs.end)
    )


def fix_random_seed(random_seed: int):
    """
    Set the same random seed for the libraries and modules.
    Includes the ``random`` module, numpy, torch, and ``uuid4()`` function 
    defined in this file.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    # Ensure deterministic ID creation
    rd = random.Random()
    rd.seed(random_seed)


def compute_chunk_span(
    start_idx: int, 
    end_idx: int, 
    hop_size: float=0.01, 
    window_size: float=0.025, 
    ndigits: int=6, 
    span_type: Literal["narrow", "middle", "broad"] = "middle",
):
    start_base = start_idx*hop_size
    end_base = end_idx*hop_size
    if span_type == "narrow":
        # narrow
        start_offset = window_size
        end_offset = 0
    elif span_type == "middle":
        # middle
        start_offset = window_size/2
        end_offset = window_size/2
    else:
        # broad
        start_offset = 0 
        end_offset = window_size
    if start_idx == 0:
        start_offset = 0
    start = round(start_base+start_offset, ndigits=ndigits)
    end = round(end_base+end_offset, ndigits=ndigits)
    return TimeSpan(start=start, end=end)


def compute_chunk_idx_by_time(
    chunk_time: float, 
    time_type: Literal["start", "end"] = "start",
    hop_size: float=0.01, 
    window_size: float=0.025,
    span_type: Literal["narrow", "middle", "broad"] = "middle",
):
    if span_type == "narrow":
        # narrow (start: window_size, end : 0)
        offset = window_size if time_type == "start" else 0
    elif span_type == "middle":
        # middle (start: window_size/2, end : window_size/2)
        offset = window_size/2
    else:
        # broad (start: 0, end : window_size)
        offset = 0 if time_type == "start" else window_size
    idx = max(0, int(round((chunk_time-offset)*1000, ndigits=3)//(hop_size*1000)))
    return idx


def compute_chunk_idx_by_span(
    chunk_span: TimeSpan, 
    hop_size: float=0.01, 
    window_size: float=0.025, 
    span_type: Literal["narrow", "middle", "broad"] = "middle",
):
    start_idx = compute_chunk_idx_by_time(
        chunk_time=chunk_span.start,
        time_type="start",
        hop_size=hop_size,
        window_size=window_size,
        span_type=span_type
    )
    end_idx = compute_chunk_idx_by_time(
        chunk_time=chunk_span.end,
        time_type="end",
        hop_size=hop_size,
        window_size=window_size,
        span_type=span_type
    )
    return start_idx, end_idx

