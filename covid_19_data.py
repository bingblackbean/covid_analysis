from labellines import labelLines
from matplotlib.dates import date2num
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

covid19_data_file = 'data/COVID_19_data.csv'
covid19_data_df = pd.read_csv(covid19_data_file)

# get numeric statistics
print(covid19_data_df.info())
# handle the missing nan
covid19_data_df.fillna(0, inplace=True)
# group the country (to drop the province)
covid19_country_rows_df = covid19_data_df.groupby(
    ['ObservationDate', 'Country/Region']).sum()
covid19_country_rows_df.reset_index(inplace=True) # split the index to two columns
# pivot to columns
covid19_country_cols_df = pd.pivot_table(
    covid19_country_rows_df,
    index='ObservationDate',
    columns='Country/Region',
    values='Confirmed',
    fill_value=0)

# check the latest date value
latest_confirmed = covid19_country_cols_df.iloc[-1, :]
latest_confirmed = latest_confirmed.sort_values(ascending=False)
top_k = 30
plt.figure(1)
plt.subplot(111)
latest_confirmed.iloc[0:top_k].plot.bar()
plt.show()


top_counties = list(latest_confirmed.index)[0:top_k]
top_counties.reverse() # reverse to add label
top_k_country_confirmed_df = covid19_country_cols_df[top_counties]

plt.figure(2)
plt.subplot(111)
time_x = pd.Series(
    pd.to_datetime(
        top_k_country_confirmed_df.index)).dt.to_pydatetime()
for col in top_counties:
    plt.plot(time_x, top_k_country_confirmed_df[col], label=str(col),)
plt.yscale('log')
labelLines(
    plt.gca().get_lines(), xvals=(
        date2num(
            datetime(
                2020, 3, 10)), date2num(
                    datetime(
                        2020, 5, 13))))
plt.show()

# check each country increase after 100 confirmed
start_num = 100
new_start_confirmed_df = pd.DataFrame([])
for col in top_counties:
    print(col)
    mask = top_k_country_confirmed_df[col] > start_num
    new_start_col = top_k_country_confirmed_df[mask][col]
    print(new_start_col)
    new_start_col.reset_index(inplace=True, drop=True)
    #new_start_col = top_k_country_confirmed_df[top_k_country_confirmed_df[col] > start_num][[col]].reset_index(drop=True)
    new_start_confirmed_df[col] = new_start_col

plt.figure(3)
plt.subplot(111)
for col in top_counties:
    plt.plot(new_start_confirmed_df[col], label=str(col))
plt.yscale('symlog')
labelLines(plt.gca().get_lines(), xvals=(10, 30))
plt.show()


"""
calculate the death
"""
# pivot to columns
covid19_country_cols_df = pd.pivot_table(
    covid19_country_rows_df,
    index='ObservationDate',
    columns='Country/Region',
    values='Deaths',
    fill_value=0)

# check the latest date value
latest_confirmed = covid19_country_cols_df.iloc[-1, :]
latest_confirmed = latest_confirmed.sort_values(ascending=False)
top_k = 30
plt.figure(1)
plt.subplot(111)
latest_confirmed.iloc[0:top_k].plot.bar()
plt.show()


top_counties = list(latest_confirmed.index)[0:top_k]
top_counties.reverse() # reverse to add label
top_k_country_confirmed_df = covid19_country_cols_df[top_counties]

plt.figure(2)
plt.subplot(111)
time_x = pd.Series(
    pd.to_datetime(
        top_k_country_confirmed_df.index)).dt.to_pydatetime()
for col in top_counties:
    plt.plot(time_x, top_k_country_confirmed_df[col], label=str(col),)
plt.yscale('log')
labelLines(
    plt.gca().get_lines(), xvals=(
        date2num(
            datetime(
                2020, 3, 10)), date2num(
                    datetime(
                        2020, 5, 13))))
plt.show()

# check each country increase after 100 confirmed
start_num = 10
new_start_confirmed_df = pd.DataFrame([])
for col in top_counties:
    print(col)
    mask = top_k_country_confirmed_df[col] > start_num
    new_start_col = top_k_country_confirmed_df[mask][col]
    print(new_start_col)
    new_start_col.reset_index(inplace=True, drop=True)
    #new_start_col = top_k_country_confirmed_df[top_k_country_confirmed_df[col] > start_num][[col]].reset_index(drop=True)
    new_start_confirmed_df[col] = new_start_col

plt.figure(3)
plt.subplot(111)
for col in top_counties:
    plt.plot(new_start_confirmed_df[col], label=str(col))
plt.yscale('symlog')
labelLines(plt.gca().get_lines(), xvals=(1, 50))
plt.show()