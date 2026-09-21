
import json
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain


file_path = '/home/weiyibiao/DCRS-main/conv/data/inspired/train_data_processed_retrieval_bm25.jsonl'
data = []

# 打开文件并逐行读取
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if line:  # 确保行不为空
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid line: {e}")

df = pd.DataFrame(data)
df['flag_id'] = range(1, len(df) + 1)



# 将 'entity' 转换为字符串并生成稀疏矩阵
df['entity_str'] = df['entity'].apply(lambda x: ' '.join(map(str, x)) if x else '')

# 使用 CountVectorizer 将 'entity' 列转为稀疏矩阵
vectorizer = CountVectorizer()
entity_matrix = vectorizer.fit_transform(df['entity_str'])

# 计算余弦相似度
similarity_matrix = cosine_similarity(entity_matrix)

# 为每行找到三个最相似的行（不包括相同 conv_id 的行）
mm_contexts = []
mm_resps = []
retrieved_mm_context_entity =[]
retrieved_mm_response_entity =[]


for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Finding most similar rows"):
    # 如果 'entity' 为空，则直接填充 None 并跳过该行
    if not row['entity']:
        mm_contexts.append([None, None, None])
        mm_resps.append([None, None, None])
        retrieved_mm_context_entity.append([None, None, None])
        retrieved_mm_response_entity.append([None, None, None])
        continue
    
    # 排除相同 conv_id 的行
    valid_indices = df[df['conv_id'] != row['conv_id']].index
    similarities = similarity_matrix[i, valid_indices]

    # 找到相似度最大的三个行
    if similarities.size > 2:  # 确保至少有三个匹配项
        best_match_indices = valid_indices[similarities.argsort()[-3:]]  # 前三大相似度索引

        # 将三个最相似的 context 和 resp 作为列表存储
        mm_contexts.append([df.at[idx, 'context'] for idx in best_match_indices])
        mm_resps.append([df.at[idx, 'un_mask_utt'] for idx in best_match_indices])
        retrieved_mm_context_entity.append([df.at[idx, 'entity'] for idx in best_match_indices]) 
        retrieved_mm_response_entity.append([df.at[idx, 'rec'] for idx in best_match_indices]) 
    else:
        # 如果不足三个匹配项，填充 None
        mm_contexts.append([None, None, None])
        mm_resps.append([None, None, None])
        retrieved_mm_context_entity.append([None, None, None])
        retrieved_mm_response_entity.append([None, None, None])

# 将每个子列表转换为字符串
processed_data = [
    [" ".join(item) if item is not None else None for item in sublist] if isinstance(sublist, list) else sublist
    for sublist in mm_contexts
]

# 将第二维和第三维合并
processed_retrieved_mm_context_entity = [list(chain.from_iterable(sublist)) if any(isinstance(i, list) for i in sublist) else sublist for sublist in retrieved_mm_context_entity]
processed_retrieved_mm_response_entity = [list(chain.from_iterable(sublist)) if any(isinstance(i, list) for i in sublist) else sublist for sublist in retrieved_mm_response_entity]


# 添加合并后的匹配结果列
df['mm_contexts'] = processed_data
df['mm_resps'] = mm_resps
df['retrieved_mm_context_entity'] = processed_retrieved_mm_context_entity
df['retrieved_mm_response_entity'] = processed_retrieved_mm_response_entity
# 手动保存为 JSONL，避免自动转义
output_path = '/home/weiyibiao/DCRS-main/conv/data/inspired/train_data_process.jsonl'
with open(output_path, 'w', encoding='utf-8') as file:
    for _, row in df.iterrows():
        json.dump(row.to_dict(), file, ensure_ascii=False)
        file.write('\n')
print(f"Data saved to {output_path} in JSONL format.")






"""
from itertools import chain
import json
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 加载 train 数据
train_file_path = '/home/weiyibiao/DCRS-main/conv/data/inspired/train_data_processed_retrieval_bm25.jsonl'
train_data = []

with open(train_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if line:
            try:
                train_data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid line: {e}")

train_df = pd.DataFrame(train_data)

# 加载 test 和 valid 数据
test_file_path = '/home/weiyibiao/DCRS-main/conv/data/inspired/test_data_processed_retrieval_bm25.jsonl'
valid_file_path = '/home/weiyibiao/DCRS-main/conv/data/inspired/valid_data_processed_retrieval_bm25.jsonl'

test_data = []
with open(test_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if line:
            try:
                test_data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid line: {e}")

valid_data = []
with open(valid_file_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if line:
            try:
                valid_data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid line: {e}")

test_df = pd.DataFrame(test_data)
valid_df = pd.DataFrame(valid_data)

# 将 'entity' 转换为字符串并生成稀疏矩阵
train_df['entity_str'] = train_df['entity'].apply(lambda x: ' '.join(map(str, x)) if x else '')
test_df['entity_str'] = test_df['entity'].apply(lambda x: ' '.join(map(str, x)) if x else '')
valid_df['entity_str'] = valid_df['entity'].apply(lambda x: ' '.join(map(str, x)) if x else '')

# 使用 CountVectorizer 将 train 的 'entity' 列转为稀疏矩阵
vectorizer = CountVectorizer()
train_entity_matrix = vectorizer.fit_transform(train_df['entity_str'])

# 对 test 和 valid 的 'entity' 计算相似度
test_entity_matrix = vectorizer.transform(test_df['entity_str'])
valid_entity_matrix = vectorizer.transform(valid_df['entity_str'])

# 计算余弦相似度
test_similarity_matrix = cosine_similarity(test_entity_matrix, train_entity_matrix)
valid_similarity_matrix = cosine_similarity(valid_entity_matrix, train_entity_matrix)

# 定义函数来查找最相似的 context 和 resp
def find_top_matches(similarity_matrix, reference_df):
    mm_contexts = []
    mm_resps = []
    retrieved_mm_context_entity =[]
    retrieved_mm_response_entity =[]

    for similarities in similarity_matrix:
        best_match_indices = similarities.argsort()[-3:]  # 前三大相似度索引

        # 将三个最相似的 context 和 resp 作为列表存储
        mm_contexts.append([reference_df.iloc[idx]['context'] for idx in best_match_indices])
        mm_resps.append([reference_df.iloc[idx]['un_mask_utt'] for idx in best_match_indices])

        retrieved_mm_context_entity.append([reference_df.iloc[idx]['entity'] for idx in best_match_indices]) 
        retrieved_mm_response_entity.append([reference_df.iloc[idx]['rec'] for idx in best_match_indices]) 


    return mm_contexts, mm_resps, retrieved_mm_context_entity,retrieved_mm_response_entity

# 获取 test 和 valid 的最相似内容
test_mm_contexts, test_mm_resps, test_retrieved_mm_context_entity, test_retrieved_mm_response_entity = find_top_matches(test_similarity_matrix, train_df)
valid_mm_contexts, valid_mm_resps,valid_retrieved_mm_context_entity , valid_retrieved_mm_response_entity= find_top_matches(valid_similarity_matrix, train_df)


# 将每个子列表转换为字符串
processed_data_test = [
    [" ".join(item) if item is not None else None for item in sublist] if isinstance(sublist, list) else sublist
    for sublist in test_mm_contexts
]

# 将每个子列表转换为字符串
processed_data_valid = [
    [" ".join(item) if item is not None else None for item in sublist] if isinstance(sublist, list) else sublist
    for sublist in valid_mm_contexts
]

retrieved_mm_context_entity_test = [list(chain.from_iterable(sublist)) if any(isinstance(i, list) for i in sublist) else sublist for sublist in test_retrieved_mm_context_entity]
retrieved_mm_response_entity_test = [list(chain.from_iterable(sublist)) if any(isinstance(i, list) for i in sublist) else sublist for sublist in test_retrieved_mm_response_entity]

retrieved_mm_context_entity_valid = [list(chain.from_iterable(sublist)) if any(isinstance(i, list) for i in sublist) else sublist for sublist in valid_retrieved_mm_context_entity]
retrieved_mm_response_entity_valid = [list(chain.from_iterable(sublist)) if any(isinstance(i, list) for i in sublist) else sublist for sublist in valid_retrieved_mm_response_entity]




# 将匹配结果添加到 test 和 valid DataFrame 中
test_df['mm_contexts'] = processed_data_test
test_df['mm_resps'] = test_mm_resps
test_df['retrieved_mm_context_entity'] = retrieved_mm_context_entity_test
test_df['retrieved_mm_response_entity'] = retrieved_mm_response_entity_test



valid_df['mm_contexts'] = processed_data_valid
valid_df['mm_resps'] = valid_mm_resps
valid_df['retrieved_mm_context_entity'] = retrieved_mm_context_entity_valid
valid_df['retrieved_mm_response_entity'] = retrieved_mm_response_entity_valid



# 将结果保存为 JSONL 格式
test_output_path = '/home/weiyibiao/DCRS-main/conv/data/inspired/test_data_process.jsonl'
valid_output_path = '/home/weiyibiao/DCRS-main/conv/data/inspired/valid_data_process.jsonl'

test_df.to_json(test_output_path, orient="records", lines=True, force_ascii=False)
valid_df.to_json(valid_output_path, orient="records", lines=True, force_ascii=False)

print(f"Test data saved to {test_output_path} in JSONL format.")
print(f"Value data saved to {valid_output_path} in JSONL format.")
"""













































# 'retrieved_contexts' 为空的行数：11658
# 'mm_contexts' 等于 [None, None, None] 的行数：11588


"""
import pandas as pd

# 加载 JSONL 文件
input_path = '/home/weiyibiao/DCRS-main/conv/data/redial/processed_data.jsonl'
df = pd.read_json(input_path, orient="records", lines=True)

# 统计 'retrieved_contexts' 列为空的行数
empty_retrieved_contexts_count = df['retrieved_contexts'].apply(lambda x: len(x) == 0).sum()

print(f"'retrieved_contexts' 为空的行数：{empty_retrieved_contexts_count}")

# 使用 apply 方法逐行检查 mm_contexts 是否等于 [None, None, None]
none_mm_contexts_count = df['mm_contexts'].apply(lambda x: x == [None, None, None]).sum()

print(f"'mm_contexts' 等于 [None, None, None] 的行数：{none_mm_contexts_count}")
"""













