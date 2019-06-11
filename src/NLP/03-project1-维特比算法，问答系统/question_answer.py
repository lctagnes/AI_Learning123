import jieba
import nltk
import json
import operator
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
import scipy.spatial.distance as distance
from sklearn import feature_extraction  
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer 

stopWords = stopwords.words("english")
#预处理 数据清洗 停用词 小写 去标点
def clean_words(strWords):
    wordList = nltk.word_tokenize(strWords)
    lemmatizer = WordNetLemmatizer()
    filteredWords = [lemmatizer.lemmatize(word.lower()) for word in wordList if word.isalpha() and word.lower() not in stopWords]
    return filteredWords
    
def read_corpus(filePath):
    """
    读取给定的语料库，并把问题列表和答案列表分别写入到 qlist, alist 里面。 在此过程中，不用对字符换做任何的处理（这部分需要在 Part 2.3里处理）
    qlist = ["问题1"， “问题2”， “问题3” ....]
    alist = ["答案1", "答案2", "答案3" ....]
    务必要让每一个问题和答案对应起来（下标位置一致）
    """
    qlist = []
    alist = []
    qlist_keyword = []# 问题的关键词列表
    with open(filePath) as file:
        json_text = file.read()
        json_dict = json.loads(json_text)
        
    for data in json_dict["data"]:
            for paragraphs in data["paragraphs"]:
                for qas in paragraphs["qas"]:
                    qlist.append(qas['question'])
                    alist.append(qas["answers"])
                    qlist_keyword.append(clean_words(qas['question']))

    assert len(qlist) == len(alist)  # 确保长度一样
    return qlist, alist,qlist_keyword

qlist, alist, qlist_keyword = read_corpus("data/train-v2.0.json")
# 统计一下在qlist 总共出现了多少个单词？
#       这里需要做简单的分词，对于英文我们根据空格来分词即可，其他过滤暂不考虑（只需分词）
def word_count(qlist):
    questions = []#分词后列表
    set_que = []
    word_total = 0
    print(len(qlist))
    for index in range(len(qlist)):
        questions.append(qlist[index].split())
    for index in range(len(questions)):
        word_total += len(questions[index])
    print(word_total)
    return word_total, questions

word_total,questions = word_count(qlist)
# 统计一下在qlist 总共出现了多少个不同的单词？
def wordSet_count(ques):
    # 定义一个集合qSet用来存储整个列表的所有单词，无重复
    qSet = set([])
    # 遍历整个数据集的每个单词word
    for word in ques:
        qSet = qSet | set(word)  
    return list(qSet)

# 统计一下qlist中每个单词出现的频率，并把这些频率排一下序，然后画成plot. 比如总共出现了总共7个不同单词，而且每个单词出现的频率为 4, 5,10,2, 1, 1,1
# 把频率排序之后就可以得到(从大到小) 10, 5, 4, 2, 1, 1, 1. 然后把这7个数plot即可（从大到小）
# 需要使用matplotlib里的plot函数。y轴是词频
# 定义一个字典计数word_count   
def sorted_word(questions,isReverse):
    word_count = {}
    word_prob = []
    for index in range(len(questions)): 
        for word in range(len(questions[index])):
            if questions[index][word] not in word_count.keys():word_count[ques[index][word]] = 0
            word_count[ques[index][word]] += 1
#     print(word_count)
    # print(len(qlist))

    for key in word_count:
        word_count[key] = word_count[key]/word_total
#     print(word_count)
    if(isReverse):
        sorted_word = sorted(word_count.items(), key=operator.itemgetter(1), reverse=True)
    else:
        sorted_word = sorted(word_count.items(), key=operator.itemgetter(1), reverse=False)
    print(sorted_word)
    for index in range(len(sorted_word)):
        word_prob.append(sorted_word[index][1])
    return sorted_word,word_prob

sorted_word,word_prob = sorted_word(True)

def plot_show():
    ## 设置字符集，防止中文乱码
    mpl.rcParams['font.sans-serif']=[u'simHei']
    mpl.rcParams['axes.unicode_minus']=False
    t=np.arange(len(word_prob))
    plt.figure(facecolor='w')#建一个画布，facecolor是背景色
    plt.plot(t, word_prob, 'r', linewidth=1, label='词频')
    plt.legend(loc = 'upper right')#显示图例，设置图例的位置
    plt.title("每个单词出现的频率", fontsize=15)
    plt.grid(b=False)#加网格
    plt.show()

#在qlist和alist里出现次数最多的TOP 10单词分别是什么？ 
def top10():
    words = []
    for index in range(10):
        words.append(word_count[index][0])
    return words

# 把qlist中的每一个问题字符串转换成tf-idf向量, 转换之后的结果存储在X矩阵里。 X的大小是： N* D的矩阵。 
# 这里N是问题的个数（样本个数），D是字典库的大小。 
        
def tfidf(qlist):
    # 每一个问题字符串转换成tf-idf向量
    vectorizer = TfidfVectorizer(smooth_idf=False, lowercase=True, stop_words=stopWords)
    # 得到的是csr_matrix型矩阵（压缩后的稀疏矩阵）
    vectorizer.fit_transform(qlist)
    # 获取词列表
    keywordList = vectorizer.get_feature_names()
    return keywordList

wordlist = tfidf(qlist)
print(wordlist)

# 矩阵X有什么特点？ 计算一下它的稀疏度
def calculate_sparse(keywordList):
    wordNum = len(keywordList)
    # 获取question总数
    docNum = len(qlist)
    #print(docNum)
    # 计算矩阵大小
    matrixSize = wordNum * docNum
    #print(matrixSize)
    # 计算零元素个数
    zeroElementNum = 0
    for question in qlist:
        for tmpWord in keywordList:
            if tmpWord not in question:
                zeroElementNum += 1

    # 根据tf-idf公式，若tf为0，那么其tf-idf值必然为零 tf-idf矩阵的稀疏度为
    return zeroElementNum / matrixSize

sparseDeg = calculate_sparse(wordlist)
print(sparseDeg)  # 打印出稀疏度

# 两个问题之间的相似度(余弦相似度计算)
def cosine_similarity(input_q, que_dict):
    simi_dict = {}
    vectorizer = TfidfVectorizer(smooth_idf=False, lowercase=True, stop_words=stopWords)
    for index, question in que_dict.items():
        tfidf = vectorizer.fit_transform([input_q, question])
        simi_value = ((tfidf * tfidf.T).A)[0, 1]
        if simi_value > 0:
            simi_dict[index] = simi_value
    return simi_dict

# 给定用户输入的问题 input_q, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
#     1. 对于用户的输入 input_q 首先做一系列的预处理，然后再转换成tf-idf向量（利用上面的vectorizer)
#     2. 计算跟每个库里的问题之间的相似度
#     3. 找出相似度最高的top5问题的答案

def top5results(input_q):
    que_dict = {}
    for index, question in enumerate(qlist):
        que_dict[index] = question
    simi_dict = cosine_similarity(input_q, que_dict)
    d = sorted(simi_dict, key=simi_dict.get, reverse=True)
    #print(d)
    # Top5最相似问题和对应的答案
    print("Top5相似-基于余弦相似度")
    for index in d[:5]:
        print("问题： " + qlist[index])
        print("答案： " + alist[index])


#基于倒排表的优化。在这里，我们可以定义一个类似于hash_map, 比如 inverted_index = {}， 然后存放包含每一个关键词的文档出现在了什么位置，
#也就是，通过关键词的搜索首先来判断包含这些关键词的文档（比如出现至少一个），然后对于candidates问题做相似度比较。 
def invert_idxTable(qlist_kw):  # 定一个简单的倒排表
    invertTable = {}
    for idx, tmpLst in enumerate(qlist_kw):
        for kw in tmpLst:
            if kw in invertTable.keys():
                invertTable[kw].append(idx)
            else:
                invertTable[kw] = [idx]
    return invertTable
# 计算倒排表
invertTable = invert_idxTable(qlist_keyword) 
#给定用户输入的问题 input_q, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
#    1. 利用倒排表来筛选 candidate
#    2. 对于用户的输入 input_q 首先做一系列的预处理，然后再转换成tf-idf向量（利用上面的vectorizer)
#    3. 计算跟每个库里的问题之间的相似度
#    4. 找出相似度最高的top5问题的答案
def filter_questionByInvertTab(inputq_keyword, qlist, invertTable):
    idx_lst = []
    q_dict = {}
    for kw in inputq_keyword:
        if kw in invertTable.keys():
            idx_lst.extend(invertTable[kw])
    idxSet = set(idx_lst)
    for idx in idxSet:
        q_dict[idx] = qlist[idx]
    return q_dict

def top5results_invidx(input_q):
    inputq_keyword = clean_words(input_q)
    filtered_qdict = filter_questionByInvertTab(inputq_keyword, qlist, invertTable)
    # 计算相似度
    simi_dict = cosine_similarity(input_q, filtered_qdict)
    d = sorted(simi_dict, key=simi_dict.get, reverse=True)
    #print(d)
    # Top5最相似问题，及它们对应的答案
    print("Top5相似-基于倒排表")
    for idx in d[:5]:
        print("问题： " + qlist[idx])
        print("答案： " + alist[idx])


# 基于词向量的文本表示
#读取每一个单词的嵌入。这个是 D*H的矩阵，这里的D是词典库的大小， H是词向量的大小。 这里面我们给定的每个单词的词向量，那句子向量怎么表达？
# 其中，最简单的方式 句子向量 = 词向量的平均（出现在问句里的）， 如果给定的词没有出现在词典库里，则忽略掉这个词。
#给定用户输入的问题 input_q, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
#     1. 利用倒排表来筛选 candidate
#     2. 对于用户的输入 input_q，转换成句子向量
#     3. 计算跟每个库里的问题之间的相似度
#     4. 找出相似度最高的top5问题的答案

def top5results_emb(input_q):
    def get_vectorValue(keywordList):
        filePath = "./data/glove.6B/glove.6B.100d.txt"
        vectorValueList = []
        with open(filePath, 'r', encoding='UTF-8') as file:
            for line in file.readlines():
                tmpLst = line.rstrip('\n').split(" ")
                word = tmpLst[0]
                if word in keywordList:
                    vectorValueList.append([float(x) for x in tmpLst[1:]])
        # 按关键词的平均，算句子的向量
        vectorSum = np.sum(vectorValueList, axis=0)
        return vectorSum / len(vectorValueList)
    
    inputq_kw = clean_words(input_q)
    # input Question中的keywords
    input_question_vector = get_vectorValue(inputq_kw)
    simi_dict = {}
    filtered_qdict = filter_questionByInvertTab(inputq_kw, qlist, invertTable)
    for idx, question in filtered_qdict.items():
        # 取得当前问题的Vector值
        filtered_question_vector = get_vectorValue(clean_words(question))
        # 计算与输入问句的cos similarity
        simi_dict[idx] = 1 - distance.cosine(input_question_vector, filtered_question_vector)

    d = sorted(simi_dict, key=simi_dict.get, reverse=True)
    print(d)
    # Top5最相似问题对应的答案
    print("计算Top5相似-基于词向量及倒排表")
    for idx in d[:5]:
        print("问题：" + qlist[idx])
        print("答案：" + alist[idx])