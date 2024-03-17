import datetime as dt
import pandas as pd
from typing import Dict, List

from task1 import get_price_from_date

def calculate_contract_value(injec_date: Dict[dt.date, float],
                             wdraw_date: Dict[dt.date, float],
                             gas_prices: pd.DataFrame,
                             rate: float,
                             max_storage_volume: float,
                             storage_cost: float) -> float:
    """
    Calculate the value of a natural gas storage contract. 
    Assume no transport delay, interest rates are zero and no holidays.
    Args:
    injec_date: the injection date of the gas and the volume of gas injected
    wdraw_date: the withdrawal date of the gas and the volume of gas withdrawn
    gas_prices: the prices at which the commodity can be purchased/sold on those dates
    rate: the cost rate at which the gas can be injected/withdrawn from the storage
    max_storage_volume: the maximum volume of gas that can be stored
    storage_cost: the cost of storing the gas, per unit volume
    """
    # create an ordered sequence of dates of change in gas volume
    # positive volume means injection, negative means withdrawal
    action_sequence = []
    for date, volume in injec_date.items():
        action_sequence.append((date, volume))
    for date, volume in wdraw_date.items():
        action_sequence.append((date, -volume))
    # sort by date
    action_sequence.sort(key=lambda x: x[0])

    # filter out invalid inject/withdraw actions due to insufficient storage volume or negative volume
    action_sequence = remove_invalid_actions(action_sequence, max_storage_volume)

    # calculate the value of the contract
    value = 0
    stored_volume = 0
    for date, volume in action_sequence:
        # calculate the injection/withdrawl cost of the action
        if volume > 0:
            i_w_cost = volume * rate
        else:
            i_w_cost = volume * rate
        # calculate the value of the gas on the date
        price = get_price_from_date(date, gas_prices)
        # calculate the value of the gas on the date
        # if volume is positive we are buying, so we subtract the value as a cost
        # if volume is negative we are selling, so we add the value as a revenue
        value -= price * volume
        # also include the injection/withdrawl cost
        value -= i_w_cost
        # calculate the storage cost
        stored_volume += volume
        value -= storage_cost * stored_volume

    return value


def remove_invalid_actions(action_sequence: List[dt.date, float], max_storage_volume: float) -> List[dt.date, float]:
    stored_volume = 0
    invalid_dates = []
    for date, volume in action_sequence:
        stored_volume += volume
        if stored_volume < 0:
            print(f"Invalid action: {volume} volume withdrawn on {date} due to no enough stored volume")
            print(f"Action removed at date {date}")
            invalid_dates.append(date)
            # redo the volume calculation
            stored_volume -= volume
        elif stored_volume > max_storage_volume:
            print(f"Invalid action: {volume} volume injected on {date} due to exceeding the maximum storage volume")
            print(f"Action removed at date {date}")
            invalid_dates.append(date)
            # redo the volume calculation
            stored_volume -= volume

    # remove invalid dates
    for date in invalid_dates:
        action_sequence = [x for x in action_sequence if x[0] != date]
    return action_sequence
