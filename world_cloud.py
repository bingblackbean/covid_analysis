import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from wordcloud import ImageColorGenerator
from wordcloud import WordCloud
import jieba

# read the data
line_list_data_cleaned_file = 'covid_analysis_book/data/COVID19_line_list_data_cleaned.csv'
line_list_data_raw_df = pd.read_csv(line_list_data_cleaned_file)

# prepare the text by using str.cat
all_symptoms = line_list_data_raw_df['symptom'].str.cat(sep=',')
print(type(all_symptoms))

# fast show wordcloud
wordcloud = WordCloud().generate(all_symptoms)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
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

print(wordcloud.words_)
print(wordcloud.process_text(all_symptoms))

# modern wordcloud
mask = np.array(Image.open('covid_analysis_book/data/human_body_shape.png'))
print(mask)
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






mask = np.array(Image.open('covid_analysis_book/data/robot.png'))
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
# refer to https://www.datacamp.com/community/tutorials/wordcloud-python
print(mask)
mask[mask == 0] = 255
image_colors = ImageColorGenerator(mask)
wordcloud = WordCloud(
    background_color="white",
    min_font_size=10,
    width=800,
    height=400,
    mask=mask,
    collocations=False,
    contour_width=2,
    contour_color='blue').generate(all_summary)
plt.figure()
# plt.imshow(wordcloud)
plt.imshow(
    wordcloud.recolor(
        color_func=image_colors),
    interpolation="bilinear")
plt.axis("off")
plt.show()




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
font = 'data/SourceHanSerifCN-Light.otf'
wordcloud = WordCloud(
    min_font_size=10,
    width=800,
    height=400,
    collocations=False,font_path=font).generate(jieba_processing_txt(ciyun_str))
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()