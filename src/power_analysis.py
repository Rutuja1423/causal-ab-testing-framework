import numpy as np
import statsmodels.stats.api as sms
from statsmodels.stats.power import TTestIndPower, zt_ind_solve_power

class ExperimentPowerAnalysis:
    """
    Class for conducting power analysis to determine required sample
    sizes for A/B tests to prevent underpowered experiments.
    
    Example usage:
        analyzer = ExperimentPowerAnalysis()
        # For a 10% baseline converting to 10.5% (5% relative lift)
        n = analyzer.calculate_sample_size_proportions(0.10, 0.05)
        print(n) # roughly 55,000 per variant
    """
    def __init__(self, alpha: float = 0.05, power: float = 0.8):
        """
        :param alpha: Significance level (Type I error rate, default 0.05)
        :param power: Statistical power (1 - Type II error rate, default 0.8)
        """
        self.alpha = alpha
        self.power = power

    def calculate_sample_size_proportions(self, p_baseline: float, relative_mde: float) -> int:
        """
        Calculate sample size per variant for a conversion rate (proportion) metric.
        
        :param p_baseline: Baseline conversion rate (e.g., 0.10 for 10%)
        :param relative_mde: Minimum detectable effect relative to baseline (e.g., 0.05 for 5% relative lift)
        :return: Required sample size per variant
        """
        if p_baseline <= 0 or p_baseline >= 1:
            raise ValueError("Baseline conversion rate must be between 0 and 1 exclusive.")

        p_target = p_baseline * (1 + relative_mde)
        
        # Calculate effect size using statsmodels proportion_effectsize
        effect_size = sms.proportion_effectsize(p_baseline, p_target)
        
        sample_size = zt_ind_solve_power(
            effect_size=effect_size,
            nobs1=None,
            alpha=self.alpha,
            power=self.power,
            ratio=1.0, # 50/50 split
            alternative='two-sided'
        )
        return int(np.ceil(sample_size))

    def calculate_sample_size_continuous(self, mu_baseline: float, std_baseline: float, relative_mde: float) -> int:
        """
        Calculate sample size per variant for a continuous metric (e.g., revenue per user).
        
        :param mu_baseline: Baseline mean (e.g., $10.0)
        :param std_baseline: Standard deviation of the baseline metric
        :param relative_mde: Minimum detectable effect relative to baseline (e.g., 0.05 for 5% relative lift)
        :return: Required sample size per variant
        """
        if std_baseline <= 0:
            raise ValueError("Standard deviation must be strictly positive.")

        absolute_mde = mu_baseline * relative_mde
        effect_size = absolute_mde / std_baseline
        
        power_analysis = TTestIndPower()
        sample_size = power_analysis.solve_power(
            effect_size=effect_size,
            nobs1=None,
            alpha=self.alpha,
            power=self.power,
            ratio=1.0,
            alternative='two-sided'
        )
        return int(np.ceil(sample_size))
