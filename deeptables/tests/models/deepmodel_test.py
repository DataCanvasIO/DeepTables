from deeptables.models.deepmodel import IgnoreCaseDict


def test_ignore_case_dict():
    metrics = {"AUC": 0.82, 'Accuracy': 0.9}
    _dict = IgnoreCaseDict(metrics)

    assert _dict['auc'] == 0.82
    assert _dict['Auc'] == 0.82
    assert _dict['AUC'] == 0.82

    assert _dict.get('AUC') == 0.82

    assert _dict['Accuracy'] == 0.9
    assert _dict['accuracy'] == 0.9
