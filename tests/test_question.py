import pytest
import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.task_manager import (
    load_heart_data, explore_data, split_data, train_random_forest,
    evaluate_model, get_feature_importances, plot_feature_importances,
    tune_n_estimators, compare_dt_vs_rf, get_oob_score, predict_patient,
    cross_validate_rf, simplify_with_top_features, run_pipeline,
)

FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


# Yardımcı — eğitilmiş model + split (testler için)
def _trained():
    X, y = load_heart_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_random_forest(X_train, y_train, n_estimators=100)
    return model, X, y, X_train, X_test, y_train, y_test


# 1. load_heart_data
def test_load_heart_data_types():
    X, y = load_heart_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_load_heart_data_shape():
    X, y = load_heart_data()
    assert X.shape == (303, 13)
    assert len(y) == 303


def test_load_heart_data_columns():
    X, y = load_heart_data()
    assert set(X.columns) == set(FEATURES)
    assert 'target' not in X.columns


def test_load_heart_data_target_binary():
    X, y = load_heart_data()
    assert set(y.unique()) == {0, 1}


# 2. explore_data
def test_explore_data_structure():
    X, y = load_heart_data()
    info = explore_data(X, y)
    assert info['n_samples'] == 303
    assert info['n_features'] == 13
    # Dengeli sınıf: 164 sağlıklı / 139 hasta
    assert info['positive_count'] == 139
    assert info['negative_count'] == 164
    assert set(info['feature_names']) == set(FEATURES)


# 3. split_data
def test_split_data_sizes():
    X, y = load_heart_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    # %20 test → 61 örnek, 242 train
    assert X_train.shape[0] == 242
    assert X_test.shape[0] == 61
    assert X_train.shape[1] == 13


def test_split_data_stratified():
    X, y = load_heart_data()
    _, _, y_train, y_test = split_data(X, y)
    full_rate = (y == 1).mean()
    test_rate = (y_test == 1).mean()
    # Stratify ile test oranı genel orana yakın olmalı
    assert abs(test_rate - full_rate) < 0.05


# 4 + 5. train_random_forest + evaluate_model
def test_train_random_forest_type():
    from sklearn.ensemble import RandomForestClassifier
    model, *_ = _trained()
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 100


def test_evaluate_model_structure():
    model, _, _, _, X_test, _, y_test = _trained()
    result = evaluate_model(model, X_test, y_test)
    for key in ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix']:
        assert key in result
    assert result['confusion_matrix'].shape == (2, 2)
    # Heart disease orta zorluk — RF iyi skor vermeli
    assert result['accuracy'] > 0.75


# 6. get_feature_importances
def test_get_feature_importances_sorted():
    model, X, *_ = _trained()
    imps = get_feature_importances(model, list(X.columns), top_n=5)
    assert len(imps) == 5
    # Azalan sırada olmalı
    vals = [v for _, v in imps]
    assert vals == sorted(vals, reverse=True)
    # Tüm önemli feature isimleri geçerli olmalı
    for name, _ in imps:
        assert name in FEATURES


def test_get_feature_importances_meaningful():
    model, X, *_ = _trained()
    top = [name for name, _ in get_feature_importances(model, list(X.columns), top_n=5)]
    # Klinik olarak anlamlı feature'lar en üstte olmalı
    # (cp=göğüs ağrısı, thalach=max kalp hızı, ca, thal, oldpeak güçlü prediktörler)
    strong = {'cp', 'thalach', 'ca', 'thal', 'oldpeak', 'age'}
    assert len(set(top) & strong) >= 3


# 7. plot_feature_importances
def test_plot_feature_importances_returns_figure():
    import matplotlib
    matplotlib.use("Agg")
    model, X, *_ = _trained()
    fig = plot_feature_importances(model, list(X.columns))
    from matplotlib.figure import Figure
    assert isinstance(fig, Figure)


# 8. tune_n_estimators
def test_tune_n_estimators():
    model, _, _, X_train, X_test, y_train, y_test = _trained()
    result = tune_n_estimators(X_train, X_test, y_train, y_test, [10, 50, 100])
    assert set(result.keys()) == {10, 50, 100}
    for acc in result.values():
        assert 0 < acc <= 1


# 9. compare_dt_vs_rf
def test_compare_dt_vs_rf():
    model, _, _, X_train, X_test, y_train, y_test = _trained()
    result = compare_dt_vs_rf(X_train, X_test, y_train, y_test)
    assert 'decision_tree' in result
    assert 'random_forest' in result
    # RF tek ağaçtan en az kadar iyi (genellikle daha iyi/kararlı)
    assert result['random_forest'] >= result['decision_tree']


# 10. get_oob_score
def test_get_oob_score():
    _, _, _, X_train, _, y_train, _ = _trained()
    oob = get_oob_score(X_train, y_train, n_estimators=100)
    assert isinstance(oob, float)
    assert 0.6 < oob < 1.0


# 11. predict_patient
def test_predict_patient_list():
    model, _, _, _, X_test, _, _ = _trained()
    sample = X_test.iloc[0].tolist()
    result = predict_patient(model, sample)
    assert result['prediction'] in [0, 1]
    assert result['risk'] in ['yüksek', 'düşük']
    assert 0 <= result['probability'] <= 1
    # prediction ile risk tutarlı olmalı
    if result['prediction'] == 1:
        assert result['risk'] == 'yüksek'
    else:
        assert result['risk'] == 'düşük'


def test_predict_patient_dict():
    model, _, _, _, X_test, _, _ = _trained()
    patient = X_test.iloc[0].to_dict()
    result = predict_patient(model, patient)
    assert 'prediction' in result
    assert 'probability' in result


# 12. cross_validate_rf
def test_cross_validate_rf():
    X, y = load_heart_data()
    result = cross_validate_rf(X, y, cv=5)
    assert 'mean' in result
    assert 'std' in result
    assert 0.7 < result['mean'] < 1.0
    assert result['std'] >= 0


# 13. simplify_with_top_features
def test_simplify_with_top_features():
    model, X, _, X_train, X_test, y_train, y_test = _trained()
    result = simplify_with_top_features(
        X_train, X_test, y_train, y_test, list(X.columns), top_n=5)
    assert 'all_features_acc' in result
    assert 'top5_acc' in result
    # Sade model (5 feature) tam modele yakın olmalı — fark çok büyük değil
    assert abs(result['all_features_acc'] - result['top5_acc']) < 0.15


# 14. run_pipeline
def test_run_pipeline_full():
    result = run_pipeline()
    for key in ['n_samples', 'test_accuracy', 'top_feature', 'oob_score',
                'dt_vs_rf_diff']:
        assert key in result
    assert result['n_samples'] == 303
    # Heart disease orta zorluk
    assert result['test_accuracy'] > 0.75
    # En kritik faktör klinik olarak anlamlı olmalı
    assert result['top_feature'] in FEATURES
    # RF tek ağaçtan daha iyi → fark pozitif (veya en azından negatif değil)
    assert result['dt_vs_rf_diff'] >= 0


# ──────────────────────────────────────────────────────
# Kaizu skor gönderimi — bu kısma DOKUNMA
# ──────────────────────────────────────────────────────

import requests


def _send_score(user_score):
    """Kaizu API'sine skor gönder. user_id ve project_id kaizu_config'ten gelir."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from kaizu_config import USER_ID, PROJECT_ID
    except ImportError:
        print("⚠️  kaizu_config.py bulunamadı — skor gönderilmeyecek.")
        return

    if USER_ID == 0:
        print("⚠️  kaizu_config.py'de USER_ID=0 — kendi ID'ni yazmadın, skor gönderilmeyecek.")
        return

    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {
        "user_id": USER_ID,
        "project_id": PROJECT_ID,
        "user_score": user_score,
        "is_auto": True,
    }
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code in (200, 201):
            print(f"✅ Skor gönderildi: {user_score}")
        else:
            print(f"⚠️  Skor gönderilemedi (HTTP {r.status_code})")
    except Exception as e:
        print(f"⚠️  Skor gönderilirken hata: {e}")


class _ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def run_tests():
    """Tüm testleri çalıştır + skoru Kaizu'ya gönder."""
    collector = _ResultCollector()
    pytest.main([os.path.dirname(__file__), "-q"], plugins=[collector])
    total = collector.passed + collector.failed
    if total == 0:
        print("Hiç test çalışmadı.")
        return
    user_score = round((collector.passed / total) * 100, 2)
    print(f"\n📊 Toplam başarılı : {collector.passed}/{total}")
    print(f"📊 Skor            : {user_score}")
    _send_score(user_score)


if __name__ == "__main__":
    run_tests()
