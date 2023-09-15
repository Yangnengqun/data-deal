

# file1过滤掉含有file2行



with open('/home/data/yang_file/data_deal/txt/val.txt', 'r') as file1:
    with open('/home/data/yang_file/data_deal/txt/test_filter_train.txt', 'r') as file2:
        lines_file1 = file1.readlines()
        lines_file2 = file2.readlines()
print(lines_file1)
# 去除包含在test1.txt中的行
filtered_lines = [line for line in lines_file1 if not any(text.strip() in line for text in lines_file2)]
# 将结果写入新的文件
with open('output_val.txt', 'w') as output_file:
    output_file.writelines(filtered_lines)