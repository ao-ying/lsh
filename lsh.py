import numpy as np
import random
from time import time
#add something new
'''
文件说明
shringle_data.txt 文档名+这个文档包含的单词构成的集合。其中后四位之前表示博主id，后四位是文章id，表示这是某个博主的第几篇文章。
word_dict.txt 词典，单词的索引和对应的单词
candidate.txt 候选对（存的是文档的标号）
candidate_article_name.txt 候选对（存的是文档名本身）
'''


# 读取数据，获得词向量。
# data[]中每一行为：文章的索引 + {包含的单词的索引}
def read_data():
    data = []
    with open('shringle_data.txt', 'r') as f:
        for line in f.readlines():
            txt = line.split()
            blog = txt[0]
            # 只读取长度大于10的文章
            if len(txt) <= 12: continue
            idx = set([int(x) for x in txt[2:]])
            data.append([blog, idx])# [string, set]
            # # 只读取一部分
            # if len(data)==1000:
            #     break
    return data

# 将词向量转化为签名向量，这是min_hash的过程
# b,r的含义为，将签名矩阵分成b个band，每个band包含r行。
def create_signature(data, b, r):
    # 生成素数，这个素数应该大于单词总个数753472，而不是大于文档总个数229354
    P = 753497  # 比753472大的最小素数
    n = b * r

    # 初始化签名矩阵
    N = len(data)  # 文档个数
    Signature_Matrix = np.zeros((n, N))

    # 进行n次打乱，构建签名向量（待优化）
    for i in range(n):  # 生成每个文档对应的签名向量的第i个元素
        # 每次打乱生成一次a和b
        a = random.randint(1, P)
        b = random.randint(1, P)
        for j in range(N):  # 对第j个文档进行操作
            Signature_Matrix[i][j] = min([(a * x + b) % P for x in data[j][1]])
    return Signature_Matrix

# lsh过程。首先生成签名矩阵，之后把签名矩阵按行分为r个band，每个band包含b行。
# 对每个band中的
def lsh(b, r, Signature_Matrix):
    #candidate_pair第i个元素代表，与文档i构成候选对的文档组成的集合。
    candidate_pair = [set() for i in range(Signature_Matrix.shape[1])]
    # 对签名矩阵进行min-hash
    start, end = 0, r #第[start,end)行，为一个band。
    while end <= (b * r):  # 签名向量每r个元素一个band，共b个band
        # 存储对每个band进行hash后得到对结果。
        local_hashBucket = {} #这是对每个band哈希之后得到的字典，之后会把结果统一到total_hashBuckets中。
        valid_key = [] #记录包含两个及两个以上文档的桶对应的键。
        for document_id in range(Signature_Matrix.shape[1]):  # 遍历签名矩阵每一列，也就是遍历每一个文档

            # 这里对每一个band使用的哈希函数是对应band中所有元素的和。
            hash_value = sum(Signature_Matrix[start:end, document_id])
            #这里哈希函数满足，当两个向量每个元素都相同时，这两个向量会被哈希到一个桶里。
            # band = Signature_Matrix[start:end, document_id]
            # hash_value = ""
            # for x in band: hash_value = hash_value + str(x) + "+" #hash值是所有数字相加得到的字符串。

            if hash_value not in local_hashBucket:
                local_hashBucket[hash_value] = [document_id]
            else :
                local_hashBucket[hash_value].append(document_id)
                #如果一个桶中的文档个数大于或等于2，则将相应的键加到valid_key中
                if len(local_hashBucket[hash_value])==2 :
                    valid_key.append(hash_value)

        #更新candidate_pair。将在valid_key中的键，对应的值添加到total_hashBuckets中
        for key in valid_key :
            vector = local_hashBucket[key]
            for i in range(len(vector)):
                temp = vector[:i] + vector[i+1:]
                candidate_pair[vector[i]] = candidate_pair[vector[i]] | set(temp)

        start += r
        end += r

    return candidate_pair

# 计算候选集中每对文档之间对相似度，并输出结果。
def calculate_similarity(candidate_pair,b ,r ,Signature_Matrix):
    result = [] #每个元素为[文档1，文档2，相似度]

    #这里x和y是一个相似对。
    for x in range(len(candidate_pair)):
        if len(candidate_pair[x]) != 0:
            for y in candidate_pair[x]:
                # 两个文档间对相似度，等于两个文档对应对签名矩阵中相等的元素所占比例。
                count = 0
                for i in range(b * r):
                    if Signature_Matrix[i][x] == Signature_Matrix[i][y]:
                        count += 1
                similarity = count / (b * r)
                result.append([x, y, similarity])
    print("Total number of candidate pair:%d" % len(result))

    # 输出结果
    #result.sort(key=lambda x: x[2], reverse=True)
    #result_distribution[i]存相似度在[i/10,i/10+0.1)之间的文档对
    result_distribution = np.zeros(10)
    for x in result:
        if x[2]==1 : result_distribution[9] += 1
        else : result_distribution[int(x[2]*10)] += 1

    print("Similarity distribution:")
    for i in range(len(result_distribution)) :
        print("similarity:%.2f～%.2f, number:%d" % (i/10,i/10+0.1,result_distribution[i]))

if __name__ == "__main__":
    # 开始计时
    s = time()

    # 设置参数
    b = 8
    r = 5
    print("b=%d, r=%d" % (b,r))
    print("Hash function: sum()")
    # 读取数据，获得词向量。
    data = read_data()
    print("Total number of documents:%d\n" % len(data))
    # min-hash,构建签名矩阵
    Signature_Matrix = create_signature(data, b, r)
    #e = time();print("Create Signature_Matrix cost:%.2fs" % (e - s))
    # 对签名矩阵进行lsh，形成候选对
    candidate_pair = lsh(b, r, Signature_Matrix)
    #e = time();print("Create candidate pair cost:%.2fs" % (e - s))
    # 计算候选集中的文档之间的相似度，输出结果。
    calculate_similarity(candidate_pair,b ,r ,Signature_Matrix)
    #e = time();print("Calculate similarity cost:%.2fs" % (e - s))

    # 结束计时
    e = time()
    # 输出总耗时
    print("\nTotal cost:%.2fs" % (e - s))



