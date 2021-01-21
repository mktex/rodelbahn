import re

url_regex = r"http[s+]?:[s+]?//(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"


def byte_2_str(text):
    if isinstance(text, bytes):
        return text.decode('utf-8')
    else:
        return text


def get_urls(xtext):
    try:
        search_result = re.findall(url_regex, byte_2_str(xtext))
    except:
        import traceback
        traceback.print_exc()
        print("Fehler in get_urls():", xtext)
        print(type(xtext))
        input("...")
        return []
    return search_result
