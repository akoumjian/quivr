import string
from typing import List, Literal, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import quivr as qv


class LargeQuivrTable(qv.Table):
    # a = qv.Int32Column()
    # b = qv.Int64Column()
    # c = qv.Float32Column()
    # d = qv.Float64Column()
    e = qv.StringColumn()
    # f = qv.BinaryColumn()
    # g = qv.BooleanColumn()
    # q = qv.LargeStringColumn()
    # r = qv.LargeBinaryColumn()


def random_for_type(type_: np.dtype, count: int) -> Union[np.ndarray, List]:
    """
    Generate a random value of a given type
    """
    if type_ == pa.int32():
        return np.random.randint(-(2**31), 2**31 - 1, count)
    elif type_ == pa.int64():
        return np.random.randint(-(2**63), 2**63 - 1, count)
    elif type_ == pa.float64():
        return np.random.normal(size=count)
    elif type_ == pa.float32():
        return np.random.normal(size=count).astype(np.float32)
    elif type_ == pa.string():
        return list(np.random.choice(list(string.ascii_letters), count))
    elif type_ == pa.binary():
        return np.random.choice(list(string.ascii_letters), count).astype(np.bytes_)
    elif type_ == pa.bool_():
        return np.random.choice([True, False], count)
    elif type_ == pa.large_string():
        return list(np.random.choice(list(string.ascii_letters), count).astype(np.str_))
    elif type_ == pa.large_binary():
        return list(np.random.choice(list(string.ascii_letters), count).astype(np.bytes_))
    else:
        raise ValueError(f"Unknown type {type_}")


@pytest.fixture
def large_table() -> LargeQuivrTable:
    """
    A pyarrow table with > 2GB of data consisting of multiple chunks
    """
    size = 100_000_000
    yield LargeQuivrTable.from_kwargs(
        a=random_for_type(pa.int32(), size),
        b=random_for_type(pa.int64(), size),
        c=random_for_type(pa.float32(), size),
        d=random_for_type(pa.float64(), size),
        e=random_for_type(pa.string(), size),
        f=random_for_type(pa.binary(), size),
        g=random_for_type(pa.bool_(), size),
        q=random_for_type(pa.large_string(), size),
        r=random_for_type(pa.large_binary(), size),
    )


@pytest.fixture
def small_table() -> LargeQuivrTable:
    """
    A pyarrow table with > 2GB of data consisting of multiple chunks
    """
    size = 100_000
    yield LargeQuivrTable.from_kwargs(
        a=random_for_type(pa.int32(), size),
        b=random_for_type(pa.int64(), size),
        c=random_for_type(pa.float32(), size),
        d=random_for_type(pa.float64(), size),
        e=random_for_type(pa.string(), size),
        f=random_for_type(pa.binary(), size),
        g=random_for_type(pa.bool_(), size),
        q=random_for_type(pa.large_string(), size),
        r=random_for_type(pa.large_binary(), size),
    )


def in_memory_sort(table: pa.Table, sort_keys: List[Tuple[str, Literal["ascending", "descending"]]]):
    """
    Sort a table in memory
    """
    indices = pc.sort_indices(table, sort_keys=sort_keys)
    table = pc.take(table, indices)
    return table

def test_sort_on_disk(small_table):
    # First test a simple single column sort
    sort_column = "e"
    sorted_table = qv.sort_on_disk(small_table, [(sort_column, "ascending")]))
    # Sort column a in memory and test that it matches the on disk sort
    sorted_column_a = in_memory_sort(small_table.table, [(sort_column, "ascending")]).column(sort_column)
    assert sorted_column_a.equals(sorted_table.column(sort_column)), (
        f"{sorted_column_a.to_pylist()}\n" f"{sorted_table.column('a').to_pylist()}"
    )


@pytest.mark.benchmark(group="sort")
def test_sort_on_disk_benchmark(benchmark, large_table):
    # First test a simple single column sort
    sort_column = "e"
    sorted_table = benchmark(qv.sort_on_disk, large_table, [(sort_column, "ascending")])
    # Sort column a in memory and test that it matches the on disk sort
    sorted_column_a = in_memory_sort(large_table.table, [(sort_column, "ascending")]).column(sort_column)
    assert sorted_column_a.equals(sorted_table.column(sort_column)), (
        f"{sorted_column_a.to_pylist()}\n" f"{sorted_table.column('a').to_pylist()}"
    )

@pytest.mark.benchmark(group="sort")
def test_sort_in_memory_benchmark(benchmark, large_table):
    # First test a simple single column sort
    sort_column = "e"
    sorted_table = benchmark(in_memory_sort, large_table.table, [(sort_column, "ascending")])
