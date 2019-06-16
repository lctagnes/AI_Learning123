# Part 1: 搭建一个分词工具
# Part 1.1 基于枚举方法来搭建中文分词工具

# TODO: 第一步： 从dic.txt中读取所有中文词。
#  hint: 思考一下用什么数据结构来存储这个词典会比较好？ 要考虑我们每次查询一个单词的效率。 
dic_words = ["北京","的","天","气","天气","真","好","真好","啊","真好啊", 
             "今","今天","课程","内容","有","很","很有","意思","有意思","课",
             "程","经常","意见","意","见","有意见","分歧","分", "歧"]    # 保存词典库中读取的单词

# 以下是每一个单词出现的概率。为了问题的简化，我们只列出了一小部分单词的概率。 在这里没有出现的的单词但是出现在词典里的，统一把概率设置成为0.00001

word_prob = {"北京":0.03,"的":0.08,"天":0.005,"气":0.005,"天气":0.06,"真":0.04,"好":0.05,"真好":0.04,"啊":0.01,"真好啊":0.02, 
             "今":0.01,"今天":0.07,"课程":0.06,"内容":0.06,"有":0.05,"很":0.03,"很有":0.04,"意思":0.06,"有意思":0.005,"课":0.01,
             "程":0.005,"经常":0.08,"意见":0.08,"意":0.01,"见":0.005,"有意见":0.02,"分歧":0.04,"分":0.02, "歧":0.005}

# 最大匹配算法的分词
def max_match(input_str):
    max_len = 5
    input_len = len(input_str) 
    segment = []
    while input_len > 0:
        word = input_str[0:max_len]
        while word not in dic_words:
            if len(word) == 1:
                break
            word = word[0:len(word)-1]
        segment.append(word)
        input_str = input_str[len(word):]
        input_len = len(input_str) 
    return segment


segment = max_match("北京的天气真好啊")
print(segment)

# 基于动态规划的分词
# dp[1] = True 意味着S1 in words
# dp[2] = True 意味着:dp[1] = True and S2 in words 或者 S1S2 in words
# dp[i] = True 意味着:dp[i-1] = True and Si in words 或者 dp[i-2] = True and S(i-1)Si in words
# 或者 dp[i-3] = True and S(i-2)S(i-1) in words.....
def word_break(input_str,words):
    # 初始化dp数组全为False,若dp[i]=True则完全可分，dp[i]=False则不完全可分
    dp = [False for i in range(len(input_str)+1)]
    # dp[0] 固定可分
    dp[0] = True
    for index in range(1,len(input_str)+1):    
        for i in range(0,index):
            if dp[i] is True and input_str[i:index] in words:
                dp[index] = True
    return dp[len(input_str)]

#  分数（10）
## TODO 请编写word_segment_dp函数来实现对输入字符串的分词
"""
    1. 对于输入字符串做分词，并返回所有可行的分词之后的结果。
    2. 针对于每一个返回结果，计算句子的概率
    3. 返回概率最高的最作为最后结果
    
    input_str: 输入字符串   输入格式：“今天天气好”
    best_segment: 最好的分词结果  输出格式：["今天"，"天气"，"好"]
"""
segments = []
def word_segment_dp(input_str,strl=''):
    # TODO： 第一步： 计算所有可能的分词结果，要保证每个分完的词存在于词典里，这个结果有可能会非常多。 
    # 存储所有分词的结果。如果次字符串不可能被完全切分，则返回空列表(list)
    if word_break(input_str,dic_words):
        if len(input_str) == 0:
            segments.append(strl[1:])
        for i in range(1, len(input_str)+1):
            if input_str[:i] in dic_words:
                word_segment_dp(input_str[i:],strl+' '+input_str[:i])
    return segments   

# TODO: 第二步：循环所有的分词结果，并计算出概率最高的分词结果，并返回
def best_segment(segments):
    score = 0
    best_score = 0
    split_str = []
    best_segment = []
    
    for size in range(len(segments)):
        split_str.append(segments[size].split())

    for i in range(len(split_str)):
        for j in range(len(split_str[i])):
            score+=word_prob.get(split_str[i][j])
            if score > best_score:
                best_score = score
                best_segment = split_str[i]
    return best_segment


# test
segments = word_segment_dp("北京的天气真好啊")
print(best_segment(segments))
segments = word_segment_dp("今天的课程内容很有意思")
print(best_segment(segments))
segments = word_segment_dp("经常有意见分歧")
print(best_segment(segments))