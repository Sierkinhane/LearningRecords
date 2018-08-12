
#   next()
s = 'Inhane'
it = iter(s)
# for i in range(7):
#     print(next(it))     # raise StopIteration

class Reverse:
    '''
    Iterator for looping over a squence backwards
    '''
    def __init__(self, data):
        self.data = data
        self.index = len(data)

    def __iter__(self):
        return  self

    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index = self.index -1
        return self.data[self.index]

rev = Reverse('Inhane')
for i in rev:
    print(i)