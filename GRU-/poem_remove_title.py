# _*_ coding:utf-8 _*_
# @Time : 2021/10/15 15:23
# @Author : xupeng
# @File : poem_remove_title.py
# @software : PyCharm

path = './poetry.txt'
new_str = ''
with open(path,'r', encoding='utf-8') as f:
    for line in f:
        new_str += line.split(":",1)[1]
path2 = './new_poem.txt'
with open(path2,'w', encoding='utf-8') as f:
    f.write(new_str)
