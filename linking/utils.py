import re

urlregex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)


def is_uri(uristr: str) -> bool:
    if uristr[0] != "<" or uristr[-1] != ">":
        return False
    return re.match(urlregex, uristr[1:-1]) is not None

def add_or_append_with_dict(dic: dict, key: str, value: dict):
    if key in dic.keys():
        for k in value.keys():
            if type(value[k]) == list:
                dic[key][k].extend(value[k])
            else:
                dic[key][k].append(value[k])
    else:
        value_dict = {}
        for k in value.keys():
            value_dict[k] = [value[k]]
        dic[key] = value_dict

def add_or_append_with_list(dic: dict, key: str, value: list):
    if key in dic.keys():
        dic[key].extend(value)
    else:
        dic[key] = value

def add_or_append_with_value(dic: dict, key: str, value: object):
    if key in dic.keys():
        dic[key].append(value)
    else:
        dic[key] = [value]


def add_or_append(dic: dict, key: str, value: str or list or dict):
    if type(value) == list:
        add_or_append_with_list(dic, key=key, value=value)
    elif type(value) == dict:
        add_or_append_with_dict(dic, key=key, value=value)
    else:
        add_or_append_with_value(dic, key=key, value=value)


