import numpy as np
from typing import List
from ..io.schemas import NettingSet, Trade

def build_simulation_grid(netting_set: NettingSet, dense_frequency_days: int = None) -> np.ndarray:
    """
    Builds the simulation time grid combining trade events and optional dense points.
    Returns array of days from t=0.
    """
    event_dates = set([0]) # Always include t=0
    
    max_maturity = 0
    
    for trade in netting_set.trades:
        if trade.trade_type == "ZCB":
            event_dates.add(trade.maturity_date)
            max_maturity = max(max_maturity, trade.maturity_date)
        elif trade.trade_type == "IRS":
            from ..products.schedule import build_irs_schedule
            sched = build_irs_schedule(trade.start_date, trade.maturity_date, trade.payment_frequency)
            event_dates.update(sched.tolist())
            event_dates.add(trade.start_date)
            max_maturity = max(max_maturity, trade.maturity_date)
        else:  # pragma: no cover
            pass

    if dense_frequency_days is not None and dense_frequency_days > 0:
        current_day = 0
        while current_day < max_maturity:
            current_day += dense_frequency_days
            if current_day < max_maturity:
                event_dates.add(current_day)
                
    grid = np.array(sorted(list(event_dates)), dtype=float)
    return grid
