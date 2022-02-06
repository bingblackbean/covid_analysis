import plotly.graph_objects as go
from collections import Counter
import missingno as msno
import pandas as pd
line_list_data_file = 'covid_analysis_book/data/COVID19_line_list_data.csv'


line_list_data_raw_df = pd.read_csv(line_list_data_file)
print(line_list_data_raw_df.info())
print(line_list_data_raw_df.describe())


line_list_data_raw_df.dropna(axis=1, how='all', inplace=True)
print(f'df shape is {line_list_data_raw_df.shape}')
# missing data visualization
msno.matrix(df=line_list_data_raw_df, fontsize=16)

date_cols = [
    'reporting date',
    'symptom_onset',
    'hosp_visit_date',
    'exposure_start',
    'exposure_end']

fig = go.Figure()
for col in date_cols:
    fig.add_trace(go.Scatter(y=line_list_data_raw_df[col], name=col))
fig.show()
print(line_list_data_raw_df[date_cols].head(5))
print(line_list_data_raw_df[date_cols].tail(5))

# check the length of date
for col in date_cols:
    date_len = line_list_data_raw_df[col].astype(str).apply(len)
    date_len_ct = Counter(date_len)
    print(f'{col} datetiem length distributes as {date_len_ct}')

# handle mix time format


def mixed_dt_format_to_datetime(series, format_list):
    temp_series_list = []
    for format in format_list:
        temp_series = pd.to_datetime(series, format=format, errors='coerce')
        temp_series_list.append(temp_series)
    out = pd.concat([temp_series.dropna(how='any')
                     for temp_series in temp_series_list])
    return out


for col in date_cols:
    line_list_data_raw_df[col] = mixed_dt_format_to_datetime(
        line_list_data_raw_df[col], ['%m/%d/%Y', '%m/%d/%y'])
print(line_list_data_raw_df[date_cols].info())

# check the length of date
for col in date_cols:
    date_len = line_list_data_raw_df[col].astype(str).apply(len)
    date_len_ct = Counter(date_len)
    print(f'{col} datetiem length distributes as {date_len_ct}')


print(line_list_data_raw_df.info())

fig = go.Figure()
for col in date_cols:
    fig.add_trace(go.Scatter(y=line_list_data_raw_df[col], name=col))
fig.show()


# the report_date is latest date.
# if the hospitalize_date is missing, we can assume it equals to the report date
# then we assume patient will go to hospital after he/she constantly has symptom for few days
# the exact number can be get by calculate the mean
# similarly we can use the symptom_date to calculate the exposure_start
# let's most people will go to





# fill missing report_date
print(line_list_data_raw_df[pd.isnull(
    line_list_data_raw_df['reporting date'])].index)
print(line_list_data_raw_df['reporting date'].iloc[260:263])
line_list_data_raw_df.loc[261, 'reporting date'] = pd.Timestamp('2020-02-11')
print(line_list_data_raw_df.info())

time_delta = line_list_data_raw_df['reporting date'] - \
    line_list_data_raw_df['hosp_visit_date']
time_delta.dt.days.hist(bins=20)
line_list_data_raw_df['hosp_visit_date'].fillna(
    line_list_data_raw_df['reporting date'], inplace=True)


#fill missing symptom_onset
time_delta = line_list_data_raw_df['hosp_visit_date'] - \
    line_list_data_raw_df['symptom_onset']
time_delta.dt.days.hist(bins=20)
average_time_delta = pd.Timedelta(days=round(time_delta.dt.days.mean()))
symptom_onset_calc = line_list_data_raw_df['hosp_visit_date'] - \
    average_time_delta
line_list_data_raw_df['symptom_onset'].fillna(symptom_onset_calc, inplace=True)
print(line_list_data_raw_df.info())


#fill missing exposure_start
time_delta = line_list_data_raw_df['symptom_onset'] - \
    line_list_data_raw_df['exposure_start']
time_delta.dt.days.hist(bins=20)
average_time_delta = pd.Timedelta(days=round(time_delta.dt.days.mean()))
symptom_onset_calc = line_list_data_raw_df['symptom_onset'] - \
    average_time_delta
line_list_data_raw_df['exposure_start'].fillna(symptom_onset_calc, inplace=True)
print(line_list_data_raw_df.info())

#fill missing exposure_end
line_list_data_raw_df['exposure_end'].fillna(line_list_data_raw_df['hosp_visit_date'], inplace=True)
print(line_list_data_raw_df.info())

fig = go.Figure()
for col in date_cols:
    fig.add_trace(go.Scatter(y=line_list_data_raw_df[col], name=col))
fig.show()

line_list_data_raw_df['case_in_country'].fillna(-1, inplace=True)
print(line_list_data_raw_df.info())

print(line_list_data_raw_df['summary'].head(5))
line_list_data_raw_df['summary'].fillna('', inplace=True)

print(line_list_data_raw_df.info())
print(line_list_data_raw_df['gender'].head(5))
line_list_data_raw_df['gender'].fillna('unknown', inplace=True)


line_list_data_raw_df['age'].hist(bins=10)
line_list_data_raw_df['age'].fillna(
    line_list_data_raw_df['age'].mean(), inplace=True)
line_list_data_raw_df['age'].hist(bins=10)

print(line_list_data_raw_df['If_onset_approximated'].head(5))
line_list_data_raw_df['If_onset_approximated'].fillna(1, inplace=True)
print(line_list_data_raw_df.info())

print(line_list_data_raw_df[pd.isnull(
    line_list_data_raw_df['from Wuhan'])].index)
print(line_list_data_raw_df[['from Wuhan','country','location']].iloc[166:175])
line_list_data_raw_df['from Wuhan'].fillna(1.0,inplace=True)

symptom = Counter(line_list_data_raw_df['symptom'])
print(symptom.most_common(2)[1][0])

line_list_data_raw_df['symptom'].fillna(symptom.most_common(2)[1][0],inplace=True)

# missing data visualization
msno.matrix(df=line_list_data_raw_df, fontsize=16)


line_list_data_raw_df.to_csv('data/COVID19_line_list_data_cleaned.csv')