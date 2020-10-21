#! -*- encoding:utf-8 -*-
"""
@File    :   tc2sc.py
@Author  :   Zachary Li
@Contact :   li_zaaachary@163.com
@Dscpt   :   tranditional chinese to simplified chinese
"""
from pyhanlp import *

def from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = []
        for line in f.readlines():
            line = line.strip().split('\t')
            lines.append(line)
    return lines

def switch(lines):
    for i in range(len(lines)):
        lines[i] = lines[i][0] + '\t' + HanLP.t2s(lines[i][1]) + '\n'
    
def save_result(file_path, lines):
    with open(file_path, 'w+', encoding='utf-8') as f:
        f.writelines(lines)

if __name__ == "__main__":
    lines = from_file('data/cmn.txt')
    switch(lines)
    save_result('data2/cmn.txt', lines)