import csv
import sys 
import os
from datetime import datetime
import random
import numpy

def random_time_transform(given_date):
    # 转换为datetime对象
    date_obj = datetime.strptime(given_date, "%Y-%m-%d")
    time_str = []

    random_key = random.uniform(0,4)

    # 转换为 Month-Day-Year 格式
    time_str.append(date_obj.strftime("%B %d, %Y"))
    print("Month-Day-Year format:", mdy_format)

    # 转换为 Day-Month-Year 格式
    time_str.append(date_obj.strftime("%d %B %Y"))
    print("Day-Month-Year format:", dmy_format)

    # 转换为 Year-Month-Day 格式
    time_str.append(date_obj.strftime("%Y-%m-%d"))
    print("Year-Month-Day format:", ymd_format)

    # 转换为 Month-Day 格式
    time_str.append(date_obj.strftime("%B %d"))
    print("Month-Day format:", md_format)

    # 转换为 Day-Month 格式
    time_str.append(date_obj.strftime("%d %B"))
    print("Day-Month format:", dm_format)

    return time_str[random_key]


file_path = "train.txt"
data = []

with open(os.path.join(sys.path[0], file_path), 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file, delimiter='\t')
    for row in reader:
        if row['id'] .isdigit():
            sample = {
                'id': int(row['id']),
                'en1': row['en1'],
                'en2': row['en2'],
                'relation': row['relation'],
                'pos1': int(row['pos1']),
                'pos2': int(row['pos2']),
                'time': row['time'],
                'sent': row['sent'].replace("<t>", random_time_transform(row['time'])) 
            }
            data.append(sample)


np.savetxt('train.txt',test,fmt='%d')
# 打印解析后的数据
for sample in data:
    print(sample)