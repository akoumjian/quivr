import gc
import tempfile
from typing import Any, Iterable, List, Literal, Tuple

import pyarrow as pa
import pyarrow.compute as pc

from . import tables


def _batch_array(to_chunk: pa.Array, chunk_size: int = 100_000) -> Iterable[pa.Array]:
    """
    Generator that yields chunks of size chunk_size from sized iterable
    (Sequence).

    Parameters
    ----------
    iterable : Sequence
        Iterable to chunk.
    chunk_size : int
        Size of chunks.

    Yields
    ------
    chunk :
        Chunk of size chunk_size from to_chunk.
    """
    offset = 0
    while offset < len(to_chunk):
        yield to_chunk.slice(offset, chunk_size)
        offset += chunk_size


def sort_on_disk(
    qv_table: tables.AnyTable, sort_keys: List[Tuple[str, Literal["ascending", "descending"]]], **kwargs: Any
) -> tables.AnyTable:
    """
    Sorts a table on disk using a list of column names and sort directions.

    Takes the same options as pyarrow.compute.sort_indices.


    :param qv_table: A :class:`Table` instance to sort.
    :param sort_keys: A list of tuples of the form (column_name, sort_direction).
    :param kwargs: Additional keyword arguments to pass to :func:`pyarrow.compute.sort_indices`.
    """
    # Generate the sorted indices
    sorted_indices = pc.sort_indices(qv_table.table, sort_keys=sort_keys, **kwargs)

    # Write the table back in chunks using sorted indices as the filter
    temp_file_name = tempfile.mktemp(suffix=".parquet")
    with pa.parquet.ParquetWriter(temp_file_name, qv_table.table.schema) as writer:
        for sorted_index_chunk in _batch_array(sorted_indices, 10_000):
            chunk_table = qv_table.table.take(sorted_index_chunk)
            writer.write_table(chunk_table)

    # Explicitly delete the table and qv_table to free up memory
    qv_table_class = qv_table.__class__
    del qv_table
    gc.collect()

    # Read the table back in
    qv_table = qv_table_class.from_parquet(temp_file_name)

    return qv_table
