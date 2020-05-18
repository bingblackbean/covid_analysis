from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from labellines import labelLines
from matplotlib.dates import date2num

covid19_data_file = 'data/COVID_19_data.csv'
covid19_data_df = pd.read_csv(covid19_data_file)

# get numeric statistics
print(covid19_data_df.info())
# handle the missing nan
#covid19_data_df.fillna(0, inplace=True)
# group the country (to drop the province)
print(covid19_data_df.head(10))
covid19_country_rows_df = covid19_data_df.groupby(
    ['ObservationDate', 'Country/Region']).sum()
print(covid19_country_rows_df.index)
covid19_country_rows_df.reset_index(
    inplace=True)  # split the index to two columns
print(covid19_country_rows_df.head(5))


def analyze_latest_day(data, top_k=30, sort_by='Deaths',plot_type=None):
    cols = ['Confirmed', 'Deaths', 'Recovered']
    latest_day_all_data = pd.DataFrame([])
    for col in cols:
        all_cols_data = pd.pivot_table(
            data,
            index='ObservationDate',
            columns='Country/Region',
            values=col,
            fill_value=0)
        latest_day_col_data = all_cols_data.iloc[-1, :]
        latest_day_all_data[col] = latest_day_col_data
    if plot_type == 'stack':
        latest_day_all_data['Unrecovered'] = latest_day_all_data['Confirmed'] - \
            latest_day_all_data['Deaths'] - latest_day_all_data['Recovered']
        latest_day_all_data = latest_day_all_data.sort_values(
            by=sort_by, ascending=False)
        latest_day_all_data.drop('Confirmed', axis=1, inplace=True)
        ax = latest_day_all_data.iloc[0:top_k].plot(kind='bar',stacked=True,title='Latest day data Statistics',color = ['#03071e', '#02c39a', '#fcbf49'])
        plt.show()
    else:
        latest_day_all_data = latest_day_all_data.sort_values(
            by=sort_by, ascending=False)
        ax = latest_day_all_data.iloc[0:top_k].plot(kind='bar', stacked=False,title='Latest day data Statistics',color = ['#fcbf49', '#03071e', '#02c39a'])
        plt.show()



analyze_latest_day(covid19_country_rows_df, top_k=30, sort_by='Deaths')

analyze_latest_day(covid19_country_rows_df, top_k=30, sort_by='Confirmed',plot_type='stack')


def analyze_col(data, col='Confirmed',top_k=30,plot_type=None):
    covid19_country_cols_df = pd.pivot_table(
        data,
        index='ObservationDate',
        columns='Country/Region',
        values=col,
        fill_value=0)
    latest_day_col_data = covid19_country_cols_df.iloc[-1, :]
    latest_day_col_data = latest_day_col_data.sort_values(
        ascending=False)
    top_counties = list(latest_day_col_data.index)[0:top_k]
    top_counties.reverse()  # reverse to add label

    top_k_country_df = covid19_country_cols_df[top_counties]

    if plot_type == 'inline_label':
        fig, ax = plt.subplots()
        # create x axis label
        time_x = pd.Series(
            pd.to_datetime(
                top_k_country_df.index)).dt.to_pydatetime()
        for c in top_counties:
            ax.plot(time_x, top_k_country_df[c], label=str(c))
        ax.set_yscale('log')
        ax.set_xlabel('date')
        ax.set_ylabel('number of cases')
        ax.set_title(f'number of {col} cases')
        labelLines(
            plt.gca().get_lines(), xvals=(
                date2num(
                    datetime(
                        2020, 3, 10)), date2num(
                            datetime(
                                2020, 5, 13))))
        plt.show()
    elif plot_type == 'inline_100':
        # check each country increase after 100 confirmed
        start_num = 100
        new_start_df = pd.DataFrame([])
        for c in top_counties:
            new_start_col = top_k_country_df[top_k_country_df[c] > start_num][c].reset_index(drop=True)
            new_start_df[c] = new_start_col
        fig, ax = plt.subplots()
        for c in top_counties:
            plt.plot(new_start_df[c], label=str(c),marker = '>',markevery=[-1])
        ax.set_yscale('symlog')
        ax.set_xlabel('days after first 100 cases')
        ax.set_ylabel('number of cases')
        ax.set_title(f'number of {col} cases')
        labelLines(plt.gca().get_lines(), xvals=(5, 35))
        plt.show()

    elif plot_type == 'tail_label_100':
        start_num = 100
        new_start_df = pd.DataFrame([])
        for c in top_counties:
            new_start_col = top_k_country_df[top_k_country_df[c] > start_num][c].reset_index(drop=True)
            new_start_df[c] = new_start_col
        fig, ax = plt.subplots()
        for c in top_counties:
            ax.plot(new_start_df[c], label=str(c))
        max_value = new_start_df.max()
        max_index = new_start_df.idxmax()
        for index, value, name in zip(
            max_index.values, max_value.values, list(
                max_value.index)):
            ax.text(
                index+0.5,
                value,
                name,
                size=10,
                ha='left',
                va='center')
            # ax.text(
            #     index+5.5,
            #     value+3,
            #     f'{value:,.0f}',
            #     size=10,
            #     ha='left',
            #     va='center')
        ax.set_yscale('symlog')
        ax.set_xlabel('days after first 100 cases')
        ax.set_ylabel('number of cases')
        ax.set_title(f'number of {col} cases')
        plt.show()

    else:
        ax =top_k_country_df.plot(kind='line')
        ax.set_xlabel('date')
        ax.set_ylabel('number of cases')
        ax.set_title(f'number of {col} cases')
        plt.show()







analyze_col(covid19_country_rows_df, col='Confirmed', top_k=30)

analyze_col(covid19_country_rows_df, col='Confirmed', top_k=30,plot_type='inline_label')

analyze_col(covid19_country_rows_df, col='Confirmed', top_k=30,plot_type='inline_100')

analyze_col(covid19_country_rows_df, col='Confirmed', top_k=10,plot_type='tail_label_100')

analyze_col(covid19_country_rows_df, col='Deaths', top_k=30)

analyze_col(covid19_country_rows_df, col='Deaths', top_k=30,plot_type='inline_label')

analyze_col(covid19_country_rows_df, col='Deaths', top_k=30,plot_type='inline_100')

analyze_col(covid19_country_rows_df, col='Deaths', top_k=10,plot_type='tail_label_100')

analyze_col(covid19_country_rows_df, col='Recovered', top_k=30,plot_type='inline_100')
