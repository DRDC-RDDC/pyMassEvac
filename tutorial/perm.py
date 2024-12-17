# The definition multichoose is taken from

# @MISC {9494,
#   TITLE = {Uniquely generate all permutations of three digits that sum to a particular value?},
#    AUTHOR = {Greg Kuperberg (https://mathoverflow.net/users/1450/greg-kuperberg)},
#    HOWPUBLISHED = {MathOverflow},
#    NOTE = {URL:https://mathoverflow.net/q/9494 (version: 2009-12-21)},
#    EPRINT = {https://mathoverflow.net/q/9494},
#    URL = {https://mathoverflow.net/q/9494}
#}

def multichoose(n,k):
    if k < 0 or n < 0: 
        return "Error"
    if not k: 
        return [[0]*n]
    if not n: 
        return []
    if n == 1: 
        return [[k]]
    return [[0]+val for val in multichoose(n-1,k)] + \
        [[val[0]+1]+val[1:] for val in multichoose(n,k-1)]

total = 0
for i in range(11):
    x = multichoose(4, i)
    total = total + len(x)

# Total is the number of way individuals can be loaded when
# not considering the constraints. Next, the constraints
# on the system have to be applied - these are used to filter
# out solutions, and those remaining are valid ways to load.

print(total)
