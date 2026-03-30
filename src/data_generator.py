import numpy as np
import pandas as pd

def generate_ab_data(n_users: int = 10000, seed: int = 42, treatment_effect: float = -0.15) -> pd.DataFrame:
    """
    Generates synthetic e-commerce A/B testing data.
    The variant in this dataset is designed to perform WORSE than the control
    to demonstrate a "Do Not Launch" decision framework.
    
    Note: The coefficients used in the logistic odds calculation below are 
     purely illustrative and are not empirically calibrated to real e-commerce data.
    """
    np.random.seed(seed)

    user_ids = [f"U_{i:05d}" for i in range(1, n_users + 1)]
    groups = np.random.choice(['Control', 'Variant'], size=n_users, p=[0.5, 0.5])
    devices = np.random.choice(['Mobile', 'Desktop', 'Tablet'], size=n_users, p=[0.6, 0.3, 0.1])
    customer_types = np.random.choice(['New', 'Returning'], size=n_users, p=[0.4, 0.6])

    # Generate Cart Value (Log-normal distribution)
    cart_values = np.random.lognormal(mean=3.5, sigma=0.8, size=n_users)
    cart_values = np.round(np.clip(cart_values, 5, 500), 2)

    # Generate Time Spent
    time_spent = np.random.normal(loc=12, scale=5, size=n_users)
    time_spent = np.round(np.clip(time_spent, 1, 60), 1)

    # Introduce missing values in time_spent (approx 5%)
    missing_idx = np.random.choice(n_users, size=int(n_users * 0.05), replace=False)
    time_spent[missing_idx] = np.nan

    # Baseline conversion log-odds
    logits = -1.5 
    # Effect of Variant (configurable, defaults to a drop in conversion)
    logits += np.where(groups == 'Variant', treatment_effect, 0)
    # Effect of Device (Mobile is harder to convert)
    logits += np.where(devices == 'Mobile', -0.3, 0)
    # Effect of Customer Type (Returning convert more)
    logits += np.where(customer_types == 'Returning', 0.5, 0)
    # Effect of Cart Value
    logits -= cart_values * 0.001

    # Calculate probabilities using sigmoid
    probs = 1 / (1 + np.exp(-logits))
    converted = np.random.binomial(n=1, p=probs)

    df = pd.DataFrame({
        'user_id': user_ids,
        'group': groups,
        'device_type': devices,
        'customer_type': customer_types,
        'cart_value': cart_values,
        'time_spent_mins': time_spent,
        'converted': converted
    })

    return df
