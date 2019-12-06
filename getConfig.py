# _*_ coding: utf-8 _*_
"""
@author: Jibao Wang
@time: 2019/11/29 15:45
"""

import configparser


def get_config(config_file='config.ini'):
    parser = configparser.ConfigParser()
    parser.read(config_file, encoding='utf-8')
    # 获取各个参数，按照 key-value 的形式保存
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
    return dict(_conf_ints + _conf_strings)
