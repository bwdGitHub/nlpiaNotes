import re

# Simple regex example
r = "(hi|hello|hey)[ ]*([a-z]*)"
match = re.match(r,'Hello Rosa',flags=re.IGNORECASE)
print(match[0])

# More complex regex
# Matches strings that start with any non a-z characters, then
# "hello","yo", etc.
# then at most 3 whitespace, comma, semicolon, or colon and at most 20 other a-z
r = r"[^a-z]*([y]o|[h']?ello|ok|hey|(good[ ])?(morn[gin']{0,3}|"\
    r"afternoon|even[gin']{0,3}))[\s,;:]{1,3}([a-z]{1,20})"

re_greeting = re.compile(r,flags=re.IGNORECASE)
print(re_greeting.match("Good Morn'n Rosa")[0])
print(re_greeting.match("1234morning::foobar"))
print(re_greeting.match("Hello RosaRosaRosaRosaRosaR"))


# regex chatbot - conditional replies based on regex matching.
my_names = set([
    'rosa',
    'rose',
    'chatty',
    'chatbot',
    'bot',
    'chatterbot'
])
curt_names = set([
    'hal',
    'you',
    'u',
])
greeter_name = 'foo'
match = re_greeting.match(input())
if match:
    at_name = match.groups()[-1]
    if at_name in curt_names:
        print("Good One")
    elif at_name.lower() in my_names:
        print("Hi {}, How are you?".format(greeter_name))