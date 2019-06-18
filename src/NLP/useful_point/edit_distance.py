# 编辑距离可以用来计算两个字符串的相似度，它的应用场景很多，其中之一是拼写纠正（spell correction）。 
# 编辑距离的定义是给定两个字符串str1和str2, 我们要计算通过最少多少代价cost可以把str1转换成str2.
# 举个例子：
# 输入: str1 = "geek", str2 = "gesek" 输出: 1 插入 's'即可以把str1转换成str2
# 输入: str1 = "cat", str2 = "cut" 输出: 1 用u去替换a即可以得到str2
# 输入: str1 = "sunday", str2 = "saturday" 输出: 3
# 我们假定有三个不同的操作： 1. 插入新的字符 2. 替换字符 3. 删除一个字符。 每一个操作的代价为1.
# 基于动态规划的解法
def edit_dist(str1, str2):
    
    # m，n分别字符串str1和str2的长度
    m, n = len(str1), len(str2)
    
    # 构建二位数组来存储子问题（sub-problem)的答案 
    dp = [[0 for x in range(n+1)] for x in range(m+1)] 
      
    # 利用动态规划算法，填充数组
    for i in range(m+1): 
        for j in range(n+1): 
  
            # 假设第一个字符串为空，则转换的代价为j (j次的插入)
            if i == 0: 
                dp[i][j] = j    
              
            # 同样的，假设第二个字符串为空，则转换的代价为i (i次的插入)
            elif j == 0:
                dp[i][j] = i
            
            # 如果最后一个字符相等，就不会产生代价
            elif str1[i-1] == str2[j-1]: 
                dp[i][j] = dp[i-1][j-1] 
  
            # 如果最后一个字符不一样，则考虑多种可能性，并且选择其中最小的值
            else: 
                dp[i][j] = 1 + min(dp[i][j-1],        # Insert 
                                   dp[i-1][j],        # Remove 
                                   dp[i-1][j-1])      # Replace 
  
    return dp[m][n] 

# str1 = "geek"
# str2 = "gesek"

# dp = edit_dist(str1,str2)
# print(dp)

# 生成指定编辑距离的单词
# 给定一个单词，我们也可以生成编辑距离为K的单词列表。 
# 比如给定 str="apple"，K=1, 可以生成“appl”, "appla", "pple"...等 下面看怎么生成这些单词。 
# 还是用英文的例子来说明。 仍然假设有三种操作 - 插入，删除，替换

def generate_edit_one(str):
    """
    给定一个字符串，生成编辑距离为1的字符串列表。
    """
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(str[:i], str[i:])for i in range(len(str)+1)]
    print(splits)
    inserts = [L + c + R for L, R in splits for c in letters]
    # print(inserts)
    deletes = [L + R[1:] for L, R in splits if R]
    # print(deletes)
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    # print(replaces)
    
    return set(inserts + deletes + replaces)

# print (len(generate_edit_one("apple")))

def generate_edit_two(str):
    """
    给定一个字符串，生成编辑距离不大于2的字符串
    """
    return [e2 for e1 in generate_edit_one(str) for e2 in generate_edit_one(e1)]

print (len(generate_edit_two("apple")))