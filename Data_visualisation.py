#pie chart
import matplotlib.pyplot as plt
plt.pie(df['class'].value_counts(), labels=['positive','negative'],autopct="%0.2f")
plt.show()
#histplot
plt.figure(figsize=(12,6))
sns.histplot(df[df['class'] == 0]['num_words'])
sns.histplot(df[df['class'] == 1]['num_words'],color='red')
#pairplot
sns.pairplot(df,hue='class')
#dropped columns and outliers
dropped=df.drop(columns=['num_characters'])
df_cleaned = dropped[dropped["num_sentences"] <= 20].copy()
df_cleaned["log_num_words"] = np.log1p(df_cleaned["num_words"])
df_cleaned["log_num_sentences"] = np.log1p(df_cleaned["num_sentences"])
df_cleaned["avg_word_length"] = (
    df["num_characters"] / df["num_words"]
).fillna(0)
df_final = df_cleaned.drop(columns=["num_words", "num_sentences"])
print(df_final.head())
import seaborn as sns

sns.pairplot(
    df_final, hue="class", vars=["log_num_words", "log_num_sentences"]
)
#heatmap

sns.heatmap(df[['num_words','num_sentences']].corr(),annot=True)
from wordcloud import WordCloud
import matplotlib.pyplot as plt
text_data = " ".join(df['text'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
