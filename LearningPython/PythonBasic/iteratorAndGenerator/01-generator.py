'''
Generator
'''

def reverse(data):
    for index in range(len(data)-1, -1, -1):
        yield data[index]
        # print(1)

for char in reverse('enahnI'):
    print(char)