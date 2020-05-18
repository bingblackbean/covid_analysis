# 前言
[第一篇文章](https://juejin.im/post/5ebd92b45188256d657b5543)和[第二篇文章](https://juejin.im/post/5ebe85185188255fd54df565)我们对line list 数据集进行清洗，以及对文本内容进行词云分析。

本文中我们将要对主要的数据集covid_19_data.csv进行清洗和分析。
这个数据集包含了所有受影响的国家的确诊，死亡，治愈人数的统计信息。 有一些国家，比如中国，美国，意大利等受疫情影响比较大的国家还有各个省/州的详细信息。

一如既往，问题优先。
今天我们简单回答两个问题：
* 截止到最近的一天，各个国家的情况如何？我们可以关注前30名。
* 前30名国家的战疫历史趋势如何？


![](https://user-gold-cdn.xitu.io/2020/5/18/172242680fb44903?w=1024&h=512&f=png&s=72547)

# 导入数据
首先导入一些包，其中有两个不常见的函数模块，后续我们会涉及到。
```
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from labellines import labelLines
from matplotlib.dates import date2num
```
读取数据，输出主要信息，这点和前几篇文章的思路一致。我们的目的就是养成一些成熟的分析数据的“套路”。
```
covid19_data_file = 'data/COVID_19_data.csv'
covid19_data_df = pd.read_csv(covid19_data_file)

# get numeric statistics
print(covid19_data_df.info())
```
数据集不大，只有1.5M,因为包含的信息不多。只有简单的confirmed，deaths，recovered信息。我们看到有一些Province/State 值缺失。
```
RangeIndex: 24451 entries, 0 to 24450
Data columns (total 8 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   SNo              24451 non-null  int64  
 1   ObservationDate  24451 non-null  object 
 2   Province/State   11706 non-null  object 
 3   Country/Region   24451 non-null  object 
 4   Last Update      24451 non-null  object 
 5   Confirmed        24451 non-null  float64
 6   Deaths           24451 non-null  float64
 7   Recovered        24451 non-null  float64
dtypes: float64(3), int64(1), object(4)
memory usage: 1.5+ MB
```
# 清理省份/州的信息
输出前10条数据，我们可以看到中国的一些省份的具体信息作为样本出现在数据集中，这对于我们分析数据造成混乱。因为我们要分析的国家的信息，所以需要对数据进行处理。

```
print(covid19_data_df.head(10))
```
```
   SNo ObservationDate Province/State  ... Confirmed Deaths  Recovered
0    1      01/22/2020          Anhui  ...       1.0    0.0        0.0
1    2      01/22/2020        Beijing  ...      14.0    0.0        0.0
2    3      01/22/2020      Chongqing  ...       6.0    0.0        0.0
3    4      01/22/2020         Fujian  ...       1.0    0.0        0.0
4    5      01/22/2020          Gansu  ...       0.0    0.0        0.0
5    6      01/22/2020      Guangdong  ...      26.0    0.0        0.0
6    7      01/22/2020        Guangxi  ...       2.0    0.0        0.0
7    8      01/22/2020        Guizhou  ...       1.0    0.0        0.0
8    9      01/22/2020         Hainan  ...       4.0    0.0        0.0
9   10      01/22/2020          Hebei  ...       1.0    0.0        0.0
[10 rows x 8 columns]
```
思路很简单，我们需要将各个省份的每天的数据累加作为中国的当天的数据，对于其他国家，如果有具体的省份数据而不是国家的总和树，我们也是如此操作。datframe 提供的groupby函数 就是专门做aggreate的。 我们需要对每天同个国家的所有条目进行求和。groupby的结果是一个MultiIndex 的新Dataframe。我们可以直接调用reset_index 将其转换成两列。
打印新数据，可以看到Mainland China 的数据为535。

```
covid19_country_rows_df = covid19_data_df.groupby(
    ['ObservationDate', 'Country/Region']).sum()
print(covid19_country_rows_df.index)
covid19_country_rows_df.reset_index(
    inplace=True)  # split the index to two columns
print(covid19_country_rows_df.head(5))

```

```
MultiIndex([('01/22/2020',            'Hong Kong'),
            ('01/22/2020',                'Japan'),
            ('01/22/2020',                'Macau'),
            ('01/22/2020',       'Mainland China'),
            ('01/22/2020',          'South Korea'),
            ('01/22/2020',               'Taiwan'),
            ('01/22/2020',             'Thailand'),
            ('01/22/2020',                   'US'),
            ('01/23/2020',            'Australia'),
            ('01/23/2020',               'Brazil'),
            ...
            ('05/13/2020', 'United Arab Emirates'),
```

```
  ObservationDate  Country/Region  SNo  Confirmed  Deaths  Recovered
0      01/22/2020       Hong Kong   13        0.0     0.0        0.0
1      01/22/2020           Japan   36        2.0     0.0        0.0
2      01/22/2020           Macau   21        1.0     0.0        0.0
3      01/22/2020  Mainland China  535      547.0    17.0       28.0
4      01/22/2020     South Korea   38        1.0     0.0        0.0
```

# 重新布局数据
现有的数据布局并不是很理想，因为ObservationDate 是一列，这一列中有很多相同的日期，不方便分析。我们需要想办法重新布局，一个好的布局方式是日期作为一个维度，国家作为另一个维度，交叉处为我们要观测的数据，比如确诊数目（Confirmed）。

解决思路是采用pandas 的pivot_table 函数。这里列出关键的参数，index 是我们最终作为row 的index的数据，columns 是我们想把源数据中哪一列的作为新数据的列（很多列）。value是我们观测的值。
为了方便调用，我这里写了一个函数来进行数据转换和分析。我们转换3个透视表，然后对于每个透视表取最后一行，也就是最新的日期。最后我们将3个透视表的最新日期的数据都统一到一个数据中。

```
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
#### 未完
```
接下来我们直接plot latest_day_all_data的数据，采用柱状图。这里有几个tips：
* 我们只plot 前top_k 个国家的数据，这样图画更有条理
* 我们对'Confirmed', 'Deaths', 'Recovered'分别标识为特定的颜色，比如confirmed 橘色（警告色），deaths 黑色（庄重色），recovered（绿色）
*  颜色代码可以从[网站](https://coolors.co/palettes/trending)轻易copy 

```
# 包含在analyze_latest_day函数中
        latest_day_all_data = latest_day_all_data.sort_values(
            by=sort_by, ascending=False)
        ax = latest_day_all_data.iloc[0:top_k].plot(kind='bar', stacked=False,title='Latest day data Statistics',color = ['#fcbf49', '#03071e', '#02c39a'])
        plt.show()
```
不得不说，这个图画还不错，但是略显紧凑。每个国家对应三根柱子，x坐标略显拥挤。

![](https://user-gold-cdn.xitu.io/2020/5/18/1722447cc1424f8e?w=1920&h=1036&f=png&s=95559)
# 改进柱状图
很容易我们就想到使用stack类型的柱状图。但是我们不能轻易的把confirmed，deaths 和recovered 堆叠再一起，这样实际意义没有参考性。所以我们需要对数据进行简单处理，用Unrecovered 来取代Confirmed的信息。
```
# 包含在analyze_latest_day函数中
        latest_day_all_data['Unrecovered'] = latest_day_all_data['Confirmed'] - \
            latest_day_all_data['Deaths'] - latest_day_all_data['Recovered']
        latest_day_all_data = latest_day_all_data.sort_values(
            by=sort_by, ascending=False)
        latest_day_all_data.drop('Confirmed', axis=1, inplace=True)
        ax = latest_day_all_data.iloc[0:top_k].plot(kind='bar',stacked=True,title='Latest day data Statistics',color = ['#03071e', '#02c39a', '#fcbf49'])
        plt.show()
```
柱子明显粗了，而且多了一个信息量（unrecovered/未痊愈/治疗中)。

这样其实我们第一个问题就有了答案，最近的一天的国家数据一目了然。画面中top_k是按照Confirmed 排序的，我们也可以按照Recovered 排序。
![](https://user-gold-cdn.xitu.io/2020/5/18/17224505da607c8b?w=1917&h=1038&f=png&s=100431)

# 各个国家时间线分析
接下来我们来分析一下各个国家确诊数据的历史信息。前文已经提到我们可以通过pivot_table重新布局数据，这里我们先整理出top_k国家的数据。

```
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
```
采用dataframe的plot函数直接预览。

```
        ax =top_k_country_df.plot(kind='line')
        ax.set_xlabel('date')
        ax.set_ylabel('number of cases')
        ax.set_title(f'number of {col} cases')
        plt.show()
```
画面看着不错，颜色绚烂。但是无法看到哪个国家对应哪个颜色。而且美国（绿色）的扶摇直上，一骑绝尘，其他国家却挤在画面的底部。
![](https://user-gold-cdn.xitu.io/2020/5/18/1722458db562ffcf?w=1918&h=1005&f=png&s=180455)
下面我们考虑如何进行改善。
* 国家分类太多，我们可不可以让标签靠近每个曲线
* 可以让y轴分布更弹性化
* x轴前半部分很单调，因为2月初只有中国和几个亚洲国家在苦苦挣扎，我们可以怎么优化？
# inline标签
我们想把标签放在曲线上，这里我们提供两种方法，第一个中较为推荐。
采用一个有用的包labellines，可以解决这个问题。调用函数labelLines即可，其中参数xvals 用于设置标签放置位置的起止位置。对于x轴为时间轴的数据，需要输入datetime格式。参考以下代码可以少入坑。

对于y轴的压缩我们可以直接用log scale。

```
      fig, ax = plt.subplots()
        # create x axis label
        time_x = pd.Series(
            pd.to_datetime(
                top_k_country_df.index)).dt.to_pydatetime()
        for col in top_counties:
            ax.plot(time_x, top_k_country_df[col], label=str(col))
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
```
看起来比刚才的要人性化一点，标签都列在曲线上，并且采用同样的颜色，看起来很像trajectory （弹道）。

![](https://user-gold-cdn.xitu.io/2020/5/18/17224630df29a0f9?w=1911&h=1003&f=png&s=335973)

# 压缩X轴
很多国家都是后来才加入到抗病毒的战争中，我们可以考虑将x轴变成“加入战斗”的时间。定义加入战斗可以从确诊数为0开始。我们这里定义为确诊从100 开始，因为最开始大多数国家都是零星的输入型案例。参考一下代码，我们对于每个国家都只保留确诊数据大于100的数据。

```
        start_num = 100
        new_start_df = pd.DataFrame([])
        for col in top_counties:
            new_start_col = top_k_country_df[top_k_country_df[col] > start_num][col].reset_index(drop=True)
            new_start_df[col] = new_start_col

```
对于plot，我们不做多的修改，仅仅在线条的末端加上一个箭头，使其更像弹道（rajectory）。

```
        # check each country increase after 100 confirmed
        start_num = 100
        new_start_df = pd.DataFrame([])
        for col in top_counties:
            new_start_col = top_k_country_df[top_k_country_df[col] > start_num][col].reset_index(drop=True)
            new_start_df[col] = new_start_col
        fig, ax = plt.subplots()
        for col in top_counties:
            plt.plot(new_start_df[col], label=str(col),marker = '>',markevery=[-1])
        ax.set_yscale('symlog')
        ax.set_xlabel('days after first 100 cases')
        ax.set_ylabel('number of cases')
        ax.set_title(f'number of {col} cases')
        labelLines(plt.gca().get_lines(), xvals=(5, 35))
        plt.show()
```

这个图看起来更不错了，这样我们可以清晰的看到各个国家的确诊曲线。
![](https://user-gold-cdn.xitu.io/2020/5/18/172246ef315d61e4?w=1919&h=1005&f=png&s=366578)

同样的，我们可以画出death 和recovered的曲线。
![](https://user-gold-cdn.xitu.io/2020/5/18/1722472f0a69f926?w=1915&h=1002&f=png&s=352270)

![](https://user-gold-cdn.xitu.io/2020/5/18/1722473b7fe19d64?w=1920&h=1005&f=png&s=349813)
# bonus：尾部标签
其实，我们可以使用matplotlib库，进行曲线尾部标注（或者任意位置）。思路就是对plot特定的位置添加text。
代码与效果如下：

```
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
```

![](https://user-gold-cdn.xitu.io/2020/5/18/17224775ef48c237?w=1916&h=1002&f=png&s=187430)

# 总结
本文接着对COVID19的数据集进行了可视化分析。文中主要介绍了一些数据变形的方法，包括groupy和pivot_table 两大利器的用法。接着我们细致的解决了一些可视化图画中的细节问题，让我们的画面更加友好。比如：
* stack类型的柱状图，颜色，数据选取
* lineplot x轴起始位置的选取
* y轴的缩放
* inline（线内）标签
* 尾端marker
所有的努力就是为了让画面更清晰的反映更多的信息。

当然了，如果需要进行交互式的查看各个国家的数据，Plotly 必然是更好的选择。