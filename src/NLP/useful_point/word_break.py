# encoding=utf-8
# 基于结巴（jieba）的分词。 Jieba是最常用的中文分词工具~
import jieba

# 基于jieba的分词
seg_list = jieba.cut("努力的学习人工智能", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  

jieba.add_word("努力的")
seg_list = jieba.cut("努力的学习人工智能", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list)) 

# 判断一句话是否能够切分（被字典）
dic = set(["贪心科技", "人工智能", "教育", "在线", "专注于"])
def word_break(str):
    could_break = [False] * (len(str) + 1)

    could_break[0] = True

    for i in range(1, len(could_break)):
        for j in range(0, i):
            if str[j:i] in dic and could_break[j] == True:
                could_break[i] = True

    return could_break[len(str)] == True

assert word_break("贪心科技在线教育")==True