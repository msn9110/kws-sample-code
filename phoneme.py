import re
import json

# include optional label
_i_opt = 1
_o_opt = 1
_s_opt = [2, 1]
optionals = [['_unknown_', '_silence_'],
             ['_unknown_1', '_silence_1'],
             ['_unknown_2', '_silence_2'],
             ['_unknown_3', '_silence_3'],
             ['_unknown_', '_silence_']]
_c = 1
tops = optionals[_i_opt * _c][:_s_opt[0]] + \
       ['no_label1'] + [chr(i) for i in range(0x3105, 0x311a)]
_c += 1
mids = optionals[_i_opt * _c][:_s_opt[0]] + \
       ['no_label2'] + [chr(i) for i in range(0x3127, 0x312a)]
_c += 1
bots = optionals[_i_opt * _c][:_s_opt[0]] + \
       ['no_label3'] + [chr(i) for i in range(0x311a, 0x3127)]
_c += 1
# input converter
indexes = {k: i for i, k in enumerate(tops + mids + bots)}
rev_indexes = {indexes[k]: k for k in indexes.keys()}

with open('valid_zhuyin.txt') as f:
    valid_zhuyins = optionals[_o_opt * _c][:_s_opt[1]] + \
                    [z for z in sorted([_ for _ in f.read().split('\n') if _])]
output_labels = valid_zhuyins
# output converter
zindexes = {k: i for i, k in enumerate(output_labels)}
rev_zindexes = {zindexes[k]: k for k in zindexes.keys()}


def get_syllable(wav_label, mode=''):
    m = {'top': '1', 'mid': '2', 'bot': '3'}
    suffix = m[mode] if mode in m else ''
    def get_label(s):
        return s if len(s) > 0 else 'no_label' + suffix

    if wav_label in ['_unknown_', '_silence_']:
        return wav_label

    if mode == 'top':
        syllable = re.sub('[^\u3105-\u3119]', '', wav_label)
    elif mode == 'bot':
        syllable = re.sub('[^\u311a-\u3126]', '', wav_label)
    elif mode == 'mid':
        syllable = re.sub('[^\u3127-\u3129]', '', wav_label)
    else:
        return wav_label

    return get_label(syllable)


zelements = {z: [get_syllable(z, m)
                 for m in ['top', 'mid', 'bot']]
             for z in valid_zhuyins}
