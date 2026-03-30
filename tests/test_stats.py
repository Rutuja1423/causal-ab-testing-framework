import pytest
import numpy as np
from src.frequentist_ab import FrequentistABTesting
from src.power_analysis import ExperimentPowerAnalysis
from src.data_generator import generate_ab_data

def test_z_test_correctness():
    # Known result calculation
    # p1 = 0.5 (50/100), p2 = 0.4 (40/100)
    result = FrequentistABTesting.z_test_proportions(
        control_conversions=50, control_trials=100,
        treatment_conversions=40, treatment_trials=100
    )
    assert result['absolute_difference'] == pytest.approx(-0.1)
    assert result['relative_lift_treatment_vs_control'] == pytest.approx(-0.2)
    assert result['p_value'] < 0.2  # Should be around ~0.15

def test_ci_calculation():
    result = FrequentistABTesting.z_test_proportions(
        control_conversions=500, control_trials=1000,
        treatment_conversions=550, treatment_trials=1000
    )
    # The true difference is +0.05
    # The confidence interval should not cross 0
    assert result['ci_95_lower'] > 0
    assert result['ci_95_upper'] > result['ci_95_lower']
    assert result['is_significant'] is True

def test_power_calculation():
    power_analyzer = ExperimentPowerAnalysis(alpha=0.05, power=0.8)
    sample_size = power_analyzer.calculate_sample_size_proportions(p_baseline=0.2, relative_mde=0.05)
    # 20% baseline, looking for 5% relative lift (which is 1% absolute -> 21%).
    # This requires roughly ~25,000 users per variant.
    assert sample_size > 20000
    assert sample_size < 30000

def test_data_generator_validity():
    df = generate_ab_data(n_users=1000)
    assert len(df) == 1000
    assert 'user_id' in df.columns
    assert 'group' in df.columns
    assert df['converted'].nunique() <= 2 # 0 or 1

def test_data_generator_effect():
    df = generate_ab_data(n_users=50000, seed=42)
    control_conv = df[df['group'] == 'Control']['converted'].mean()
    variant_conv = df[df['group'] == 'Variant']['converted'].mean()
    # The variant should actively underperform the control
    assert variant_conv < control_conv
