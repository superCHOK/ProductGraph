from pythainlp.tokenize import word_tokenize

text = "ทดสอบการตัดคำภาษาไทย"
proc = word_tokenize(text, engine='newmm')
print(proc)