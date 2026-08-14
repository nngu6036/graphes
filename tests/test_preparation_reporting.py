from __future__ import annotations

from grapher.data.preparation_reporting import (
    common_preparation_report,
    print_preparation_summary,
)


def test_common_preparation_report_uses_one_count_schema() -> None:
    report = common_preparation_report(
        input_records=20,
        processed_records=15,
        accepted_graphs=12,
        rejection_reasons={"parse_failure": 3},
    )

    assert report == {
        "num_input_records": 20,
        "num_processed_records": 15,
        "num_accepted_graphs": 12,
        "num_rejected_records": 3,
        "num_unexamined_records": 5,
        "rejection_reasons": {"parse_failure": 3},
    }


def test_print_preparation_summary_has_stable_labels(capsys) -> None:
    print_preparation_summary(
        dataset="Example",
        source="source.sdf",
        input_records=20,
        processed_records=15,
        accepted_graphs=12,
        rejection_reasons={"parse_failure": 3},
        split_sizes={"train": 10, "val": 1, "test": 1},
        outputs=(("topology + attributes", "outputs/example", "node: atom"),),
    )

    assert capsys.readouterr().out.splitlines() == [
        "Prepared dataset: Example",
        "  source: source.sdf",
        "  input records: 20",
        "  processed records: 15",
        "  accepted graphs: 12",
        "  rejected records: 3",
        "  unexamined records: 5",
        "  rejection reasons: {'parse_failure': 3}",
        "  splits: train=10, val=1, test=1",
        "  topology + attributes: outputs/example (node: atom)",
    ]
