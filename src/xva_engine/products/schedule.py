import numpy as np

def year_fraction_act365f(d1: float | np.ndarray, d2: float | np.ndarray) -> float | np.ndarray:
    """
    Returns the year fraction between two dates (in days) using ACT/365 Fixed convention.
    """
    return (d2 - d1) / 365.0

def build_irs_schedule(start_date: int, maturity_date: int, payment_frequency_months: int) -> np.ndarray:
    """
    Builds an idealized payment schedule for an IRS.
    Given simple start/maturity in days from t=0.
    """
    # Approximate months as 30 days (payment frequency is in months)
    # Real logic would use actual date objects and month additions
    step_days = payment_frequency_months * 30
    
    dates = []
    current_date = start_date + step_days
    while current_date < maturity_date:
        dates.append(current_date)
        current_date += step_days
    # always close out at maturity
    dates.append(maturity_date)
    return np.array(dates)
