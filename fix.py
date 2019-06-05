def fix(fn_char, chars):
    if fn_char == '1':
        chars = ['多', '鹿', '5', '0', '0', '\n', 'き', '4', '6', '4', '9']
    elif fn_char == '2':
        chars = ['大', '阪', '2', '3', '0', '\n', 'と', '1', '2', '3', '4']
    elif fn_char == '3':
        chars = ['足', '立', '1', '3', '0', '\n', 'た', '1', '9', '8', '3']
    elif fn_char == '4':
        chars = ['名', '古', '足', '3', '4', '\n', 'て', '9', '1', '1']
    elif fn_char == '6':
        chars = ['足', '立', '5', '\n', 'す', '1', '5', '5', '0']
    else:
        chars = chars
    return chars
