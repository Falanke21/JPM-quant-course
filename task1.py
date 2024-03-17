# import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import datetime as dt


def visualization(df: pd.DataFrame):
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Dates'], df['Prices'], label='Natural Gas Prices', marker='o')

    # Formatting x-axis as months
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    # Show only every 4th label
    for label in ax.xaxis.get_ticklabels()[::4]:
        label.set_visible(True)
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    # Rotating x-axis labels for better readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add labels and title
    plt.xlabel('Date')
    plt.ylabel('Natural Gas Price')
    plt.title('Natural Gas Prices Over Time')

    # Show legend at the top left
    plt.legend(loc='upper left')

    plt.savefig('nat_gas_prices.png')


def predict_for_future(date: dt.date, df: pd.DataFrame) -> float:
    """
    Predict the price of natural gas on a given date
    between the last date in the data set and one year from that date.
    Any date outside of this range should give an error message.
    Prerequisite: date is after the last date in the dataset,
    but within one year from that date.
    """
    last_date = df['Dates'][len(df)-1]
    last_date_plus_one_year = last_date + dt.timedelta(days=365)

    if date < last_date or date > last_date_plus_one_year:
        raise ValueError("Given date is not valid. Please enter a date between {} and {}".format(
            last_date, last_date_plus_one_year))

    # predict the price
    # 1. find the year average price of each year in the dataset
    # 2. find the average year growth rate by comparing the average price of each year
    # 3. use the average year growth rate to predict the price of the given month by the formula:
    #    month = date.month
    #    year = date.year
    #    price = the same month price of 2024 * (1 + average year growth rate) ^ (2024 - year)
    # 4. return the predicted price

    # 1. find the year average price of each year in the dataset
    df['Year'] = df['Dates'].apply(lambda x: x.year)
    year_avg_price = df.groupby('Year')['Prices'].mean()
    print(f"year_avg_price: {year_avg_price}")

    # 2. find the average year growth rate by comparing the average price of each year
    year_growth_rate = year_avg_price.pct_change().mean()
    print(f"year_growth_rate: {year_growth_rate}")

    # 3. use the average year growth rate to predict the price of the given month
    month = date.month
    year = date.year
    # find the same month price of 2024
    same_month_price_2024 = year_avg_price[2024]
    price = same_month_price_2024 * (1 + year_growth_rate) ** (2024 - year)
    print(f"predicted price: {price}")
    return price


def estimate_from_history(date: dt.date, df: pd.DataFrame) -> float:
    """
    Estimate the price of natural gas on a given date
    between the first and last date in the dataset.
    Prerequisite: date is within the range of the dataset dates.
    """
    # find nearest date before the given date in the dataset
    # we will use linear search for now, if the dataset gets larger we can use binary search
    date_before = df['Dates'][df['Dates'] <= date].max()

    # find nearest date after the given date in the dataset
    date_after = df['Dates'][df['Dates'] >= date].min()

    if date_before == date_after:
        print("The given date is in the dataset. We will use the price on that date.")
        return df['Prices'][df['Dates'] == date].values[0]

    date_before_price = df['Prices'][df['Dates'] == date_before].values[0]
    date_after_price = df['Prices'][df['Dates'] == date_after].values[0]

    # find number of days between each date and the given date
    days_before = (date - date_before).days
    days_after = (date_after - date).days

    # now we can estimate the price on the given date
    # using linear interpolation
    price = date_before_price + \
        (date_after_price - date_before_price) * \
        (days_before / (days_before + days_after))
    print(f"estimated price: {price}")
    return price


def get_price_from_date(date: dt.date, df: pd.DataFrame):
    """
    Estimate the price of natural gas on a given date
    between the first last date in the dataset and 
    one year from the last date in the dataset.
    Any date outside of this range should give an error message.
    """
    first_date = df['Dates'][0]
    last_date = df['Dates'][len(df)-1]
    # print(f"first date: {first_date}")
    # print(f"last date: {last_date}")
    last_date_plus_one_year = last_date + dt.timedelta(days=365)

    if date < first_date or date > last_date_plus_one_year:
        raise ValueError("Given date is not valid. Please enter a date between {} and {}".format(
            first_date, last_date_plus_one_year))

    if date > last_date:
        print("The given date is after the last date in the dataset. The price will be predicted.")
        # predict the price
        result = predict_for_future(date, df)
    else:
        print("The given date is within the dataset. "
              "The price will be estimated.")
        # estimate the price
        result = estimate_from_history(date, df)
    return result


if __name__ == '__main__':
    # Load CSV data
    df = pd.read_csv('Nat_Gas.csv', parse_dates=['Dates'])
    # Convert df['Dates'] to python datetime date objects
    df['Dates'] = df['Dates'].dt.date
    # print the date field first row
    print(df['Dates'][0])
    new_date = dt.date(2025, 5, 26)
    print(f"new date: {new_date}")
    print(
        f"new_date is larger than the first date: {new_date > df['Dates'][0]}")
    # visualization(df)

    print(f"Estimate the price of natural gas on date: {new_date}")
    get_price_from_date(new_date, df)
