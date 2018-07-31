'''
2018年7月17日17:56:37
Regular expression operations
'''

import re

# .(dot) this matches any characters except a newline
dot_matches = re.findall(r'.', 'mind\npocket')

# matches the start of the string
caret_matches = re.findall(r'^m', 'mind pocket')

# matches the end of the string ---> ['mind pocket']
dollar_sign = re.findall(r'[a-z(\s)]+t$', 'mind pocket') 

# ab* will match any number of 'a', 'ab', or 'a' followed by any number of 'b's 
# output ---> ['ab', 'ab', 'a', 'a', 'abbbb']
star_matches = re.findall(r'ab*', 'ababaaabbbb')

# matche 1 or more repetitions.
# ab+ will match 'a' followed by any non-zero number of 'b's, it will not match just 'a'
# ['ab', 'ab', 'abbbb']
plus_matches = re.findall(r'ab+', 'ababaaabbbb')

# ab? will match either 'a' or 'ab'
question_mark = re.findall(r'ab?', 'ababaaabbbb')

# {m}
# output ---> ['bbbb']
m_matches = re.findall(r'b{4}', 'ababaaabbbb')
# output ---> ['bbb', 'bbbb'] matches either 3 'b's or 4 'b's
m2_matches = re.findall(r'b{3,4}', 'abababbbaabbbb')
# output ---> ['bbb', 'bbb'] only matches 3 'b's
m3_matches = re.findall(r'b{3,4}?', 'abababbbaabbbb')
# \
m4_matches = re.findall(r'\\', '*\\mind pocket')

# [amk] will match 'a' 'm' or 'k'
m5_matches = re.findall(r'[amk]', 'aaammlkk')
# [0-9A-Za-z] will match any hexadecimal digit
# output ---> ['a', 's', 'd', 'w', '2', '2', '3', 'A', 'S', 'D', 'F']
m6_matches = re.findall(r'[0-9A-Za-z]', 'asdw223ASDF')
# [a\-z] [-a] [a-] will match '-'
# output ---> ['-', 'a']
m7_matches = re.findall(r'[a\-z]', '-aww')

# Special characters lose their special meaning inside sets. 
# For example, [(+*)] will match any of the literal characters
# '(', '+', '*', or ')'.

# Module Contents
pattern = r'[a-b]+'
prog = re.compile(pattern)
string = 'aabbdababwdasdw'
result = prog.match(string) # is equivalent to result = re.match(pattern, string)

prog = re.compile(r'aab')
result = prog.search(string)

# split
# If maxsplit is nonzero, at most maxsplit splits occur, and the remainder of the 
# string is returned as the final element of the list.
# >>> re.split(r'\W+', 'Words, words, words.')
# ['Words', 'words', 'words', '']
# >>> re.split(r'(\W+)', 'Words, words, words.')
# ['Words', ', ', 'words', ', ', 'words', '.', '']
# >>> re.split(r'\W+', 'Words, words, words.', 1)
# ['Words', 'words, words.']
# >>> re.split('[a-f]+', '0a3B9', flags=re.IGNORECASE)
# ['0', '3', '9']
# >>> re.split(r'(\W+)', '...words, words...')
# ['', '...', 'words', ', ', 'words', '...', '']
# >>> re.split(r'\b', 'Words, words, words.')
# ['', 'Words', ', ', 'words', ', ', 'words', '.']
# >>> re.split(r'\W*', '...words...')
# ['', '', 'w', 'o', 'r', 'd', 's', '', '']
# >>> re.split(r'(\W*)', '...words...')
# ['', '...', '', '', 'w', '', 'o', '', 'r', '', 'd', '', 's', '...', '', '', '']
# sub ---> anihaohhnihaonihaonihao
result = re.sub(r'-', 'nihao', 'a-hh---')
# findall
print(result)




