# 前言
上一篇文章（[链接](https://juejin.im/post/5ebd92b45188256d657b5543)）我们对COVID19_line_list数据集进行了清洗以及初步分析。本文中我们将分析如何用词云来展示文本信息的概要。

比如我们从[词云百度百科](https://baike.baidu.com/item/%E8%AF%8D%E4%BA%91)截取文字，制作词云。简单来说，词云就是重要单词的可视化，如下图。
![](https://user-gold-cdn.xitu.io/2020/5/15/172184d18951b31b?w=1909&h=966&f=png&s=727192)
line list 数据集中有两列很重要的文本信息，symptoms （症状） 以及summary（摘要）。我们可以轻易的提出两个问题：
* COVID19 的主要症状是什么
* 文本摘要的内容主要是什么

我们将用词云回答这两个问题。

python 作为一个万能胶水语言，各种有用的轮子自然不胜枚举。[wordcloud](https://amueller.github.io/word_cloud/) 便是专门用于制作词云的包。 安装方式很简单，pip即可。

![](https://user-gold-cdn.xitu.io/2020/5/15/1721853afa4e13cc?w=1024&h=512&f=png&s=72547)
# 准备数据
数据我们采用[上篇](https://juejin.im/post/5ebd92b45188256d657b5543)中清理好的数据，这里我将清理好的数据保存为新的csv文件（COVID19_line_list_data_cleaned.csv）。

第一步，导入必要的库。
```
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from wordcloud import ImageColorGenerator
from wordcloud import WordCloud
import jieba
```

```
# read the data
line_list_data_cleaned_file = 'data/COVID19_line_list_data_cleaned.csv'
line_list_data_raw_df = pd.read_csv(line_list_data_cleaned_file)
```
我们需要分析的是symptom 和summary 两列的信息。
wordcloud 分析的文本为str 格式，因此我们需要将dataframe 每一行的数据组合成一个str 格式。

```
# prepare the text by using str.cat
all_symptoms = line_list_data_raw_df['symptom'].str.cat(sep=',')
print(type(all_symptoms))
```
```
<class 'str'>
```
可以看到all_symptoms 已经是str 格式。我们先分析symptom 列，后续会处理summary列的信息。

# 快速做经典词云
借用经典的案例代码，我们先用默认的参数制作词云。

```
# fast show wordcloud
wordcloud = WordCloud().generate(all_symptoms)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```

![](https://user-gold-cdn.xitu.io/2020/5/15/172185ad4382e307?w=1916&h=982&f=png&s=307479)
很不错，已经有了初步的模样，不过我们还是发现一些问题：
* 有些词太小了，几乎看不见
* 两个fever 是个什么东西？
* 字体好模糊，不能更清楚吗？

当然能解决，wordCould 类带有一些初始化参数，比如min_font_size控制最小的词字体大小，像素大小通过width和height 来调节。默认的collocations 为True，用于合并重要性/频次相当的文本。设定这些参数，我们可以轻而易举的改善的词云画面。
```
# change the resolution and cancel the collocations
wordcloud = WordCloud(
    min_font_size=10,
    width=800,
    height=400,
    collocations=False).generate(all_symptoms)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
```
一款经典版的词云就出炉了，看上去很不错。我们的第一个问题也有了答案： fever 和cough 是最常见的症状。
![](https://user-gold-cdn.xitu.io/2020/5/15/172185fbd6357474?w=1906&h=976&f=png&s=449283)
# 更modern的词云
这里有一幅人类体形图，我们也可以将这些症状的词条作为tag 刻画在人物肖像上。这里需要用到wordcloud的mask 参数。mask 顾名思义就是用于掩盖一些像素。

![](https://user-gold-cdn.xitu.io/2020/5/15/1721867128162644?w=515&h=958&f=png&s=45702)
加载图像，并且转化为array作为mask。print mask的信息，我们可以看到大批量的255 255 255。 这是一个好的mask，因为这个代表着白色，白色的区域我们将不会用于填写词条，仅仅对有色区域进行填写。
```
# modern wordcloud
mask = np.array(Image.open('data/human_body_shape.png'))
print(mask)
```

```
 [[255 255 255   0]
  [255 255 255   0]
  [255 255 255   0]
  ...
  [255 255 255   0]
  [255 255 255   0]
  [255 255 255   0]]
 [[255 255 255   0]
  [255 255 255   0]
  [255 255 255   0]
  ...
  [255 255 255   0]
  [255 255 255   0]
  [255 255 255   0]]
 [[255 255 255   0]
  [255 255 255   0]
  [255 255 255   0]
  ...
  [255 255 255   0]
  [255 255 255   0]
  [255 255 255   0]]]
```
再次创建wordcloud，代码几乎和上次雷同，仅仅是添加一个mask参数，以及设定图像的线条宽度contour_width 以及颜色contour_color。

```
wordcloud = WordCloud(
    background_color="white",
    min_font_size=10,
    width=800,
    height=400,
    mask=mask,
    collocations=False,
    contour_width=2,
    contour_color='black').generate(all_symptoms)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
```

![](https://user-gold-cdn.xitu.io/2020/5/15/172186d64f44296e?w=931&h=876&f=png&s=70364)
效果看起来比经典款的要好一些，但是还有一些瑕疵。我们可以看到body 轮廓中很多空白处，这是因为symptom 统计的词条类数目比较少，无法填满图像。
# 彩色图像词云
很明显，summary 列的信息量要远远大于symptom，下面我们可以分析该列数据。
这次我们选择一幅彩色图像，我把human换成robot。几乎同样的代码，再次运行。

![](https://user-gold-cdn.xitu.io/2020/5/15/172187208eb517a5?w=960&h=960&f=png&s=131832)

```
mask = np.array(Image.open('data/robot.png'))
all_summary = line_list_data_raw_df['summary'].str.cat(sep=',')
image_colors = ImageColorGenerator(mask)
wordcloud = WordCloud(
    background_color="white",
    min_font_size=10,
    width=800,
    height=400,
    mask=mask,
    collocations=False,
    contour_width=1,
    contour_color='black').generate(all_summary)
plt.figure()
plt.imshow(
    wordcloud.recolor(
        color_func=image_colors),
    interpolation="bilinear")
plt.axis("off")
plt.show()
```
结果。。。oops, 说好的机器人呢？怎么只有两个眼睛和几个大门牙，一定是mask出了问题。

![](https://user-gold-cdn.xitu.io/2020/5/15/1721874a69c8d505?w=899&h=846&f=png&s=321071)

我们打印一下创建的mask矩阵。一堆堆零，边框明明是白色的，为什么是零呢？datacamp 博客给出了一定的[解释](https://www.datacamp.com/community/tutorials/wordcloud-python)。总之，零不是我们想要的。

```
[[[0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]
  ...
  [0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]]
 [[0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]
  ...
  [0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]]
 [[0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]
  ...
  [0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]]
 ...
 [[0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]
  ...
  [0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]]
 [[0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]
  ...
  [0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]]
 [[0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]
  ...
  [0 0 0 0]
  [0 0 0 0]
  [0 0 0 0]]]

```
解决方案也很简单，替换0为255，然后重新制作词云。

```
mask[mask == 0] = 255
```
可爱的机器人终于出现了。 

回到我们开始提到的问题，我们可以看到summary主要是关于新确认的（new confirmed）一些COVID 案例，病人(patient)可能和Wuhan相关。而且我们可以看到样本中male 似乎比female 多一些。

![](https://user-gold-cdn.xitu.io/2020/5/15/172187ad85f446a3?w=924&h=853&f=png&s=180139)

到此我们的两个问题都圆满的通过词云回答了。
# bonus: 中文词云
回到开篇的词云图，我们展示了一份中文词云。如果直接借用我们今天的代码可能会出现一些问题。这里我们仅仅贴出中文词云制作的代码，以及一点注意事项。
* 处理画面出现显示异常，可能是字体的问题。
* 画面中词分割不好？ 用jieba

ciyun.csv 就是从百度词条随便截取的，你可以换成任意的文章。

```
ciyun = 'data/ciyun.csv'
with open(ciyun) as f:
    ciyun_str  = f.read()

def jieba_processing_txt(text):
    mywordlist = []
    seg_list = jieba.cut(text, cut_all=False)
    liststr = "/ ".join(seg_list)
    for myword in liststr.split('/'):
        if len(myword.strip()) > 1:
            mywordlist.append(myword)
    return ' '.join(mywordlist)
font = 'data/SourceHanSerifCN-Light.otf' # 可以下载或者用电脑的自带的字体
wordcloud = WordCloud(
    min_font_size=10,
    width=800,
    height=400,
    collocations=False,font_path=font).generate(jieba_processing_txt(ciyun_str))
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
```

# 总结
本文介绍了经典版以及画面嵌套版的词云制作。使用词云可以一目了然的获取海量文本内容的关键信息。词云制作过程中的一些坑我们也进行了掩埋：
* 画面分辨率问题
* 叠词问题
* 彩色画面的嵌套问题
* 中文乱码的问题
