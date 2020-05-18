# 前言
[上一篇](https://juejin.im/post/5ec191b1e51d454dca71174f)我们对数据进行了重新布局，布局后的数据结构方便我们进行柱状图可视化以及弹道分析。

今天我们来学习使用该数据集执着更酷炫的动态排名视频。

先看效果：

![](https://user-gold-cdn.xitu.io/2020/5/19/17229021a0de82dd?w=800&h=427&f=gif&s=312044)
一如既往，直奔代码。
![](https://user-gold-cdn.xitu.io/2020/5/19/172290332ffa69d8?w=1024&h=512&f=png&s=74683)
# 准备数据源
数据源就是我们一直分析的COVID19 data 数据，可以去kaggle 下载。

导入我们所需的库，相比于之前的文章，我们本次分析会用到animation模块，重点是里面会提供FuncAnimation 类，帮助我们实现动态图。

```
# coding: utf-8
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from datetime import datetime, timedelta
import numpy as np
```
pandas 读取数据，这是每篇分析的第一步操作。
简单处理数据，采用groupby 函数将一些国家的各个省份信息合并成该国家的总和。
前一篇文章有详细介绍，此处不再说明。

```
# read data
covid19_data_file = 'data/COVID_19_data.csv'
covid19_data_df = pd.read_csv(covid19_data_file)
# handle the countries data
df_country = covid19_data_df.groupby(
    ['ObservationDate', 'Country/Region']).sum()
df_country.reset_index(inplace=True)
```
# 动态视频思路-FuncAnimation
大家都知道，视频就是一堆堆图像（或者称为帧 frame)在时间轴上连续起来形成的。所以我们的思路也很简单，制作一个画面，改变画面的内容，重复制作这个画面。

matplotlib 已经有这个类：[FuncAnimation](https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.animation.FuncAnimation.html?highlight=funcanimation#matplotlib.animation.FuncAnimation)，它用来重复调用一个函数进行画图。我们来研究一下它的主要参数，更详细的请参考官方文档。

```
class matplotlib.animation.FuncAnimation(fig, func, frames=None, init_func=None, fargs=None, save_count=None, *, cache_frame_data=True, **kwargs)[source]¶

```
其中主要的参数：
* fig： 就是matplotlib的Figure 对象。
* func： 就是需要重复调用的函数。对于我们这个案例来说，需要重复的事情就是“画（水平）柱状图”。所以我们需要定义个画水平柱状图的函数。这也是本文的重点。
* frames： 就是可迭代的对象，假如我们赋值为整数n，就是用range(n)来创造迭代对象
* init_func： 类似于func，如果你的第一帧画面需要调用不同的函数，可选此参数
* fargs： func 函数的其他参数（除去frames 必须作为第一个位置参数）
* 其他参数：略

为了调用这个函数，我们需要准备好各个参数。
* 采用subplots 创建Figure 对象，命名为fig。
* 调用datetime，设置需要动态显示的起止日期，并且计算出delta 时间。该值我们将作为frames 参数传递给FuncAnimation函数。
* 剩下就是重中之重，func 函数以及fargs 参数

```
fig, ax = plt.subplots(figsize=(15, 8))
start_date = datetime(2020, 1, 22)
end_date = datetime(2020, 5, 13)
dates_delta = (end_date - start_date).days
```

# 每一帧画面的绘制函数func

先上代码，再做解释。

```
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
```

![](https://user-gold-cdn.xitu.io/2020/5/19/17229204c58bab82?w=1902&h=956&f=png&s=437530)
代码中我们主要实现一下内容：
* 整理数据，选出每天top10的国家，并且降序排列
* 绘制barh，水平绘制时，需要反转y轴，使得最大值排在第一位。也就是上图中第1部分内容绘制完毕
* 添加国家名称以及对应的确诊数据。也就是上图中第2 和第3部分内容
* 添加大写的日期，放在右下角，也就是图中第4部分
* 里面还有一些细节，比如取消掉y轴的标签

函数准备好了，下面我们就将函数的对应的参数传递给FuncAnimation。
```
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
```
我们也可以使用以下代码将其保存为本地mp4格式。

```
writer = animation.writers['ffmpeg']
writer = writer(fps=1)
animator.save('mini_covid_bar_race.mp4', writer=writer)
```
我们看一下上述代码的输出结果，这里我将视频转成gif以做演示。基本效果已经成型，应该算是很经典的动态排名了。

![](https://user-gold-cdn.xitu.io/2020/5/19/172292d47cefe024?w=800&h=427&f=gif&s=206307)
# 来点更炫的（彩色+动态文字+xkcd）
## 彩色柱状图
给柱状图添加颜色，应该很好处理。barh 函数带有color 参数，这里仅仅需要注意传入的颜色需要是类数组的格式。
小技巧：
* 由于我们无法为所有的国家定义颜色，因此这里我们采用定义一个dict颜色集，里面定义主要国家的颜色，然后对于没有定义在dict中的国家，颜色采用默认。颜色代码的获取可以从很多网站查询和复制。

```
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
# barh 中的color 参数为：
# color=[
#                color_dict.get(
#                    x,
#                    "#f8edeb") for x in top_k_df[cat_col]],
```
## 添加动态文字
这里我添加了一些文字来给视频做注释。比如3月15日，中国捐给西班牙50万个口罩。

* 之所以用英文，是因为最初这个视频是我放在facebook上给老外看的。

* 第二个原因，是因为中文需要一些字体支持。

实现动态文字添加的思路很简单，就是ax.text 函数。实现方法类似于我们的国家标签以及确诊数的标签。
![](https://user-gold-cdn.xitu.io/2020/5/19/172293c5fb54980d?w=1912&h=955&f=png&s=336611)

```
timeline_event = {
    '01/30/2020': 'WuHan declared lockdown.',
    '01/31/2020': 'Italian suspended all flights from China',
    '02/02/2020': 'Trump restricts on any foreigners from entering the U.S',
    '03/13/2020': 'China sent medical supplies to Italy',
    '03/15/2020': 'China donated 500,000 facemasks to Spain',
    '03/19/2020': 'USA suspended visa services worldwide.',
    '05/12/2020': 'America first(LOL).'
}
```
## 添加xkcd 效果
[xkcd](https://zh.wikipedia.org/wiki/Xkcd) 是啥？ 只不过一个漫画名称而已，不好发音，也不是缩写。对于matplotlib 来说，xkcd 就指的类似于漫画的的效果。通俗讲就是“线条抖啊~~抖啊~~抖~~~~”
代码很简单就一行：

```
    with plt.xkcd():
        把所有plt相关的代码放在这个with 里面
```

## 完整的func 函数
除了添加颜色，动态文字以及“抖啊抖”的效果，我们还做了一些细节处理，比如调整字体颜色，字号等小细节。
```
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
```
重新调用这个新的func来制作动画。

```
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
```
最后我们来看一下我们更新后的动画效果。ps. 眼看着中国从top10中消失，眼看着America First。

![](https://user-gold-cdn.xitu.io/2020/5/19/1722949944f5513e?w=800&h=427&f=gif&s=1171080)


# Tips
保存为MP4格式需要电脑安装ffmep 编码/解码器，安装好的ffmpeg_path需要添加到matplotlibrc 参数下。

```
# add ffmpeg path to matplotlibrc
plt.rcParams['animation.ffmpeg_path'] = r'your_path\ffmpeg-20200323-ba698a2-win64-static\ffmpeg-20200323-ba698a2-win64-static\bin\ffmpeg.exe'

```
# 总结
本文中我们继续使用covid19的数据来进行可视化分析。我们采用python 制作了酷炫的动态排名。
定义的函数可以套用在其他数据集中用于制作动态排名。
通过本文我们可以学会：
* 如何制作动态排名（barh race） 图，以及保存为视频
* 如何给bar 不同类别赋予不同的颜色
* 如果给画面添加文字
* 如何是画面显得“抖一抖”
# 八卦

听说Youtuber 使用Bar race 制作视频，月入50万。

