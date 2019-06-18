# 出现频率特别高的和频率特别低的词对于文本分析帮助不大，一般在预处理阶段会过滤掉。 
# 在英文里，经典的停用词为 “The”, "an"....

# 方法1： 自己建立一个停用词词典
stop_words = ["the", "an", "is", "there"]
# 在使用时： 假设 word_list包含了文本里的单词
word_list = ["we", "are", "the", "students"]
filtered_words = [word for word in word_list if word not in stop_words]
print (filtered_words)

# 方法2：直接利用别人已经构建好的停用词库
from nltk.corpus import stopwords
cachedStopWords = stopwords.words("english")

from nltk.stem.porter import *
stemmer = PorterStemmer()

test_strs = ['caresses', 'flies', 'dies', 'mules', 'denied',
         'died', 'agreed', 'owned', 'humbled', 'sized',
         'meeting', 'stating', 'siezing', 'itemization',
         'sensational', 'traditional', 'reference', 'colonizer',
         'plotted']

singles = [stemmer.stem(word) for word in test_strs]
print(' '.join(singles))  # doctest: +NORMALIZE_WHITESPACE