from __future__ import annotations

from typing import Dict


def _require_keys(dataset_name: str, example: Dict, keys) -> None:
    missing = [key for key in keys if key not in example]
    if missing:
        raise KeyError(
            f"EoE split projection for dataset `{dataset_name}` is missing required key(s): {missing}"
        )


def build_eoe_raw_example(dataset_name: str, example: Dict) -> Dict:
    name = str(dataset_name or "").strip()

    if name == "trivia_qa_split":
        _require_keys(name, example, ("question", "answer"))
        return {
            "question": example["question"],
            "answer": example["answer"],
        }

    if name == "nq_open":
        _require_keys(name, example, ("question", "answer"))
        return {
            "question": example["question"],
            "answer": example["answer"],
        }

    if name == "ambig_qa":
        _require_keys(name, example, ("question", "annotations", "nq_answer"))
        return {
            "question": example["question"],
            "annotations": example["annotations"],
            "nq_answer": example["nq_answer"],
        }

    if name == "situated_qa":
        _require_keys(name, example, ("question", "answer", "references", "ambig_label"))
        return {
            "question": example["question"],
            "answer": example["answer"],
            "references": example["references"],
            "ambig_label": example["ambig_label"],
        }

    if name == "boolq":
        _require_keys(name, example, ("question", "answer", "passage"))
        return {
            "question": example["question"],
            "answer": example["answer"],
            "passage": example["passage"],
        }

    if name == "squad_split":
        _require_keys(name, example, ("question", "answers", "context"))
        return {
            "question": example["question"],
            "answers": example["answers"],
            "context": example["context"],
        }

    if name in {"alcuna_trivia_qa", "temporalood_trivia_qa"}:
        _require_keys(name, example, ("question", "answers", "ood_label"))
        return {
            "question": example["question"],
            "answers": example["answers"],
            "ood_label": example["ood_label"],
        }

    raise NotImplementedError(
        f"EoE raw split projection is not implemented for dataset `{dataset_name}`."
    )


def eoe_feature_fn(dataset_name, tokenizer, ex):
    del tokenizer
    return build_eoe_raw_example(dataset_name, ex)
