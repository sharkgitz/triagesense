from schema import RequestType
from evaluation import compute_metrics


def test_accuracy_is_exact_on_hand_built_predictions():
    y_true = [
        "Billing/Claim Dispute",
        "Billing/Claim Dispute",
        "Claim Status/Coverage Enquiry",
        "Claim Status/Coverage Enquiry",
    ]
    y_pred = [
        "Billing/Claim Dispute",
        "Claim Status/Coverage Enquiry",
        "Claim Status/Coverage Enquiry",
        "Claim Status/Coverage Enquiry",
    ]
    result = compute_metrics(y_true, y_pred)
    assert result["accuracy"] == 0.75


def test_confusion_matrix_contains_all_five_classes_even_if_unused():
    y_true = ["Billing/Claim Dispute", "Claim Status/Coverage Enquiry"]
    y_pred = ["Billing/Claim Dispute", "Claim Status/Coverage Enquiry"]
    result = compute_metrics(y_true, y_pred)
    cm = result["confusion_matrix"]
    all_classes = [rt.value for rt in RequestType]
    assert list(cm.index) == all_classes
    assert list(cm.columns) == all_classes
    # classes never seen should be all zero, no KeyError raised getting here
    assert cm.loc[RequestType.ESCALATION.value].sum() == 0


def test_class_with_zero_predictions_has_zero_precision_and_recall_no_nan():
    y_true = ["Billing/Claim Dispute", "Billing/Claim Dispute"]
    y_pred = ["Billing/Claim Dispute", "Billing/Claim Dispute"]
    result = compute_metrics(y_true, y_pred)
    per_class = result["per_class"]

    escalation_row = per_class.loc[RequestType.ESCALATION.value]
    assert escalation_row["precision"] == 0.0
    assert escalation_row["recall"] == 0.0
    assert escalation_row["f1"] == 0.0
    assert not per_class.isna().any().any()


def test_compute_metrics_handles_empty_input_without_crashing():
    result = compute_metrics([], [])
    assert result["accuracy"] == 0.0
    assert len(result["confusion_matrix"]) == len(list(RequestType))
