# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from datetime import datetime, timedelta
import numpy as np

# add ffmpeg path to matplotlibrc
plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\renb\Downloads\ffmpeg-20200323-ba698a2-win64-static\ffmpeg-20200323-ba698a2-win64-static\bin\ffmpeg.exe'
# read data
covid19_data_file = '../data/covid_19_data.csv'
covid19_data_df = pd.read_csv(covid19_data_file)
# handle the countries data
df_country = covid19_data_df.groupby(
    ['ObservationDate', 'Country/Region']).sum()
df_country.reset_index(inplace=True)


fig, ax = plt.subplots(figsize=(15, 8))
start_date = datetime(2020, 1, 22)
end_date = datetime(2020, 5, 13)
dates_delta = (end_date - start_date).days

"""
mini bar race
"""


def mini_bar_chart_frame(
        delta,
        df=None,
        start_date=None,
        date_col=None,
        cat_col=None,
        observe_col=None,
        top_k=10,
        ax=None):
    if start_date is None:
        start_date = datetime(2020, 2, 22)
    date_show = timedelta(days=delta) + start_date
    date_str = date_show.strftime('%m/%d/%Y')
    top_k_df = df[df[date_col].eq(date_str)].sort_values(
        by=observe_col, ascending=False).head(top_k)
    ax.clear()
    # plot horizon bar
    ax.barh(
        top_k_df[cat_col],
        top_k_df[observe_col],
        log=False)
    ax.invert_yaxis()  # to make the biggest in the top
    #dx = np.log(top_k_df[observe_col].max()) / 200
    for i, (value, name) in enumerate(
            zip(top_k_df[observe_col], top_k_df[cat_col])):
        ax.text(
            value - 20,
            i,
            name,
            size=10,
            weight=600,
            ha='right',
            va='center')
        ax.text(
            value + 0.1,
            i,
            f'{value:,.0f}',
            size=10,
            ha='left',
            va='center')
    ax.text(
        1,
        0.1,
        date_str,
        transform=ax.transAxes,
        size=40,
        ha='right',
        weight=800)
    ax.set_yticks([])  # we have label on the top of bar


fargs = (df_country,
         start_date,
         'ObservationDate',
         'Country/Region',
         'Confirmed',
         10,
         ax)
animator = animation.FuncAnimation(
    fig,
    mini_bar_chart_frame,
    frames=dates_delta,
    fargs=fargs,
    interval=1000,
    repeat=False)
writer = animation.writers['ffmpeg']
writer = writer(fps=1)
animator.save('mini_covid_bar_race.mp4', writer=writer)

"""
classic bar race
"""


def bar_chart_frame(
        delta,
        df=None,
        start_date=None,
        date_col=None,
        cat_col=None,
        observe_col=None,
        top_k=10,
        color_dict=None,
        ax=None):
    print(start_date)
    if start_date is None:
        start_date = datetime(2020, 2, 22)
    print(start_date)
    date_show = timedelta(days=delta) + start_date
    date_str = date_show.strftime('%m/%d/%Y')
    top_k_df = df[df[date_col].eq(date_str)].sort_values(
        by=observe_col, ascending=False).head(top_k)
    print(top_k_df)
    ax.clear()
    # plot horizon bar
    ax.barh(
        top_k_df[cat_col],
        top_k_df[observe_col],
        color=[
            color_dict.get(
                x,
                "#f8edeb") for x in top_k_df[cat_col]],
        log=False,
        left=1)
    ax.invert_yaxis()  # to make the biggest in the top
    #dx = np.log(top_k_df[observe_col].max()) / 200
    for i, (value, name) in enumerate(
            zip(top_k_df[observe_col], top_k_df[cat_col])):
        ax.text(
            value - 20,
            i,
            name,
            size=10,
            weight=600,
            ha='right',
            va='center')
        ax.text(
            value + 0.1,
            i,
            f'{value:,.0f}',
            size=10,
            ha='left',
            va='center')
    ax.text(
        1,
        0.1,
        date_str,
        transform=ax.transAxes,
        color='#f8edeb',
        size=40,
        ha='right',
        weight=800)
    ax.text(
        0.5,
        1.1,
        'Covid-19',
        transform=ax.transAxes,
        size=14,
        color='#f8edeb')
    ax.text(
        0.2,
        0.05,
        timeline_event.get(date_str, ''),
        transform=ax.transAxes,
        size=20,
        color='#06d6a0')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    ax.set_yticks([])
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    plt.box(False)


timeline_event = {
    '01/30/2020': 'WuHan declared lockdown.',
    '01/31/2020': 'Italian suspended all flights from China',
    '02/02/2020': 'Trump restricts on any foreigners from entering the U.S',
    '03/13/2020': 'China sent medical supplies to Italy',
    '03/15/2020': 'China donated 500,000 facemasks to Spain',
    '03/19/2020': 'USA suspended visa services worldwide.',
    '05/12/2020': 'America first(LOL).'
}


color_dict = {'Mainland China': '#e63946',
              'US': '#ff006e',
              'Italy': '#02c39a',
              'Span': '#f4a261',
              'UK': '#3a86ff',
              'Germany': '#370617',
              'France': '#3a86ff',
              'Japan': '#d8e2dc',
              'Iran': '#fec89a',
              'Russia': '#dc2f02'}


fig, ax = plt.subplots(figsize=(15, 8))
start_date = datetime(2020, 1, 22)
end_date = datetime(2020, 5, 13)
dates_delta = (end_date - start_date).days
fargs = (df_country,
         start_date,
         'ObservationDate',
         'Country/Region',
         'Confirmed',
         10,
         color_dict,
         ax)
animator = animation.FuncAnimation(
    fig,
    bar_chart_frame,
    frames=dates_delta,
    fargs=fargs,
    interval=1000,
    repeat=False)
writer = animation.writers['ffmpeg']
writer = writer(fps=1)
animator.save('covid_bar_race.mp4', writer=writer)



def xkcd_bar_chart_frame(
        delta,
        df=None,
        start_date=None,
        date_col=None,
        cat_col=None,
        observe_col=None,
        top_k=10,
        color_dict=None,
        ax=None):

    if start_date is None:
        start_date = datetime(2020, 2, 22)
    date_show = timedelta(days=delta) + start_date
    date_str = date_show.strftime('%m/%d/%Y')
    top_k_df = df[df[date_col].eq(date_str)].sort_values(
        by=observe_col, ascending=False).head(top_k)
    with plt.xkcd():
        ax.clear()
        # plot horizon bar
        ax.barh(
            top_k_df[cat_col],
            top_k_df[observe_col],
            color=[
                color_dict.get(
                    x,
                    "#f8edeb") for x in top_k_df[cat_col]],
            log=False,
            left=1)
        ax.invert_yaxis()  # to make the biggest in the top
        #dx = np.log(top_k_df[observe_col].max()) / 200
        for i, (value, name) in enumerate(
                zip(top_k_df[observe_col], top_k_df[cat_col])):
            ax.text(
                value - 20,
                i,
                name,
                size=10,
                weight=600,
                ha='right',
                va='center')
            ax.text(
                value + 0.1,
                i,
                f'{value:,.0f}',
                size=10,
                ha='left',
                va='center')
        ax.text(
            1,
            0.1,
            date_str,
            transform=ax.transAxes,
            color='#f8edeb',
            size=40,
            ha='right',
            weight=800)
        ax.text(
            0.5,
            1.1,
            'Covid-19',
            transform=ax.transAxes,
            size=14,
            color='#f8edeb')
        ax.text(
            0.2,
            0.05,
            timeline_event.get(date_str, ''),
            transform=ax.transAxes,
            size=20,
            color='#06d6a0')
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
        ax.xaxis.set_ticks_position('top')
        ax.tick_params(axis='x', colors='#777777', labelsize=12)
        ax.set_yticks([])
        ax.margins(0, 0.01)
        ax.grid(which='major', axis='x', linestyle='-')
        ax.set_axisbelow(True)
        plt.box(False)



fig, ax = plt.subplots(figsize=(15, 8))
start_date = datetime(2020, 1, 22)
end_date = datetime(2020, 5, 13)
dates_delta = (end_date - start_date).days
fargs = (df_country,
         start_date,
         'ObservationDate',
         'Country/Region',
         'Confirmed',
         10,
         color_dict,
         ax)
animator = animation.FuncAnimation(
    fig,
    xkcd_bar_chart_frame,
    frames=dates_delta,
    fargs=fargs,
    interval=1000,
    repeat=False)
writer = animation.writers['ffmpeg']
writer = writer(fps=1)
animator.save('covid_bar_race.mp4', writer=writer)
