# _*_ coding: utf-8 _*_
"""
@author: Jibao Wang
@time: 2019/11/29 17:22
"""
import unicodedata
import re, io
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


# 将 unicode 文件转换成 ascii
def unicode_to_ascii(str):
    return ''.join(c for c in unicodedata.normalize('NFD', str) if unicodedata.category(c)!='Mn')
# 句子预处理，去除前后空字符，替换特殊字符，添加首尾控制符等
def preprocess_sentence(sentence):
    sentence = unicode_to_ascii(sentence.lower().strip())
    # 在单词与末尾标点符号之间添加一个空格
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # 将“非法”字符用 空格 替换
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
    sentence = sentence.rstrip().strip() # 去除首尾空格
    sentence = '<start> ' + sentence + ' <end>'  # 句子添加首尾标志
    return sentence
# 创建数据集，en 是英文句子，sp 是西班牙语
def create_dataset(file_path):
    lines = io.open(file_path, encoding='utf-8').read().strip().split('\n')
    en = []
    sp = []
    for line in lines:
        en.append(preprocess_sentence(line.strip().split('\t')[0]))
        sp.append(preprocess_sentence(line.strip().split('\t')[1]))
    return en, sp

def tokenizer(sentence_list):
    language_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    language_tokenizer.fit_on_texts(sentence_list)
    tensor = language_tokenizer.texts_to_sequences(sentence_list)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, language_tokenizer

def load_dataset(file_path):
    en_target_language, sp_input_language = create_dataset(file_path)
    sp_input_tensor, sp_input_language_tokenizer = tokenizer(sp_input_language)
    en_target_tensor, en_target_language_tokenizer = tokenizer(en_target_language)
    return sp_input_tensor, en_target_tensor, sp_input_language_tokenizer, en_target_language_tokenizer
