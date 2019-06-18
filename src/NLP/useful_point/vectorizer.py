# 词袋向量： 把文本转换成向量 。 只有向量才能作为模型的输入。

# 方法1： 词袋模型（按照词语出现的个数）
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
corpus = [
     'He is going from Beijing to Shanghai.',
     'He denied my request, but he actually lied.',
     'Mike lost the phone, and phone was in the car.',
]
X = vectorizer.fit_transform(corpus)
print (X.toarray())
print (vectorizer.get_feature_names())

# 方法2：词袋模型（tf-idf方法）
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(smooth_idf=False)
X = vectorizer.fit_transform(corpus)

print (X.toarray())
print (vectorizer.get_feature_names())