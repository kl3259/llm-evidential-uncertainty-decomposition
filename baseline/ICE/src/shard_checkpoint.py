import json
import os
import tempfile


def ensure_parent_dir(path: str):
    save_dir = os.path.dirname(path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)


def write_json_list_atomic(path: str, records):
    ensure_parent_dir(path)
    directory = os.path.dirname(path) or "."
    fd, temp_path = tempfile.mkstemp(
        dir=directory,
        prefix=".tmp_",
        suffix=os.path.basename(path) or ".json",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _load_json_list(path: str):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"Checkpoint file is not a list: {path}")
    return records


def _record_index(record, merge_key: str, source_key: str):
    if merge_key in record:
        return int(record[merge_key])
    if source_key in record:
        return int(record[source_key])
    raise KeyError(f"Record is missing both `{merge_key}` and `{source_key}`.")


def sort_records(records, merge_key: str, source_key: str):
    return sorted(records, key=lambda row: _record_index(row, merge_key, source_key))


def load_shard_checkpoint(output_path: str, skip_output_path: str, merge_key: str, source_key: str):
    output_records = sort_records(_load_json_list(output_path), merge_key, source_key)
    skip_records = sort_records(_load_json_list(skip_output_path), merge_key, source_key)

    processed_indices = set()
    for label, records in (("output", output_records), ("skip", skip_records)):
        for record in records:
            idx = _record_index(record, merge_key, source_key)
            if idx in processed_indices:
                raise ValueError(
                    f"Duplicate checkpoint index `{idx}` detected while loading {label} records."
                )
            processed_indices.add(idx)
    return output_records, skip_records, processed_indices


def write_shard_checkpoint(
    output_path: str,
    output_records,
    skip_output_path: str,
    skip_records,
    merge_key: str,
    source_key: str,
):
    write_json_list_atomic(output_path, sort_records(output_records, merge_key, source_key))
    write_json_list_atomic(skip_output_path, sort_records(skip_records, merge_key, source_key))
