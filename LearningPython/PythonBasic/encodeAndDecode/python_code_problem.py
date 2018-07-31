'''
python 编码问题
2018年7月14日17:11:19
'''

import sys
#   python3
#   1.系统默认编码设置为UTF-8。
#   2.用str类型表示文本字符，str能表示Unicode字符集中所有字符，二进制字节数据用bytes来表示。

#   获取默认编码
default_encoding = sys.getdefaultencoding() # utf-8
#print(default_encoding)

#   str类型
name = 'mindpocket'    # <class 'str'>
chinese = '中国'      # <class 'str'>
#print(type(name))
#print(type(chinese))

#   bytes类型
#   python3中，在字符引号前面加’b‘，就表示这是一个bytes类型的对象，实际上它就是一组二进制字
#   字节序列组成的数据，bytes类型可以是ASCII范围内的字符和其他十六进制形式的字符数据，但不能
#   用中文等非ASCII字符表示
name = b'mindpocket'    # <class 'bytes'>
#chinese = b'中国'     # SyntaxError: bytes can only contain ASCII literal characters.
#print(type(name))

#   encode--储存
#   decode--显示
name = 'mindpocket思想口袋'
enc = name.encode()     # b'mindpocket\xe6\x80\x9d\xe6\x83\xb3\xe5\x8f\xa3\xe8\xa2\x8b'
enc_GBK = name.encode("gbk")    # b'mindpocket\xcb\xbc\xcf\xeb\xbf\xda\xb4\xfc'
#print(enc)
#print(enc_GBK)

dec = enc.decode()    # mindpocket思想口袋
dec_GBK = enc_GBK.decode('gbk')     # mindpocket思想口袋
# print(dec)
# print(dec_GBK)

# encoding--编码或解码方式
# open() 默认使用的是本地的编码GBK，所以用GBK或者使用默认参数打开会乱码--['浣犲ソ鍟婁腑鍥斤紒']
# decode('xx',strict='ignore') ignore参数
with open('chinese.txt', encoding='utf-8') as txt_file:
    arr = [x for x in txt_file.readlines()]
# print(arr)
