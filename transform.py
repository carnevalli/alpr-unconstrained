import itertools
import re

def swapletters(lp):
    lp = lp.upper()
    zero_tuple = ('O', 'Q', 'D', '0', 'U', '8')
    one_tuple = ('1', 'I', '7', 'L')
    two_tuple = ('2', 'Z')
    five_tuple = ('5', 'S')
    six_tuple = ('6', 'G')
    find = {
        '0' : zero_tuple,
        'O' : zero_tuple,
        'Q' : zero_tuple,
        'D' : zero_tuple,
        'U' : zero_tuple,
        '8' : zero_tuple,
        '1' : one_tuple,
        'I' : one_tuple,
        '7' : one_tuple,
        'L' : one_tuple,
        '5' : five_tuple,
        'S' : five_tuple,
        '2' : two_tuple,
        'Z' : two_tuple,
        '6' : six_tuple,
        'G' : six_tuple
    }

    replacement_list = []
    for i in range(len(lp)):
        if lp[i] in find:
            replacement_list.append(find[lp[i]])
        else:
            replacement_list.append(lp[i])
            
    return list(map("".join, itertools.product(*replacement_list)))

def findsimilar(lp_str, regex_patterns):
    lp_filtered_candidates = []
    lp_candidates = swapletters(lp_str)
    for candidate in lp_candidates:
        for pattern_id, pattern in regex_patterns:
            matches = re.findall(pattern, candidate, flags=re.IGNORECASE)   
            for match in matches:
                if not (pattern_id, match) in lp_filtered_candidates:
                    lp_filtered_candidates.append((pattern_id, match))
    return lp_filtered_candidates