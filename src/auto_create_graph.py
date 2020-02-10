#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import csv
import re
import string
import nltk

from pythainlp.tokenize import Tokenizer
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import dict_trie
from pythainlp.tag import pos_tag

from nltk.corpus import words
from stop_words import get_stop_words

import webbrowser
from py2neo import Node, Relationship, Graph

file_name = 'category1'

FILE = './Data/{}.csv'.format(file_name)
df = pd.read_csv(FILE)

# Drop a column that not use
data = df.drop(columns=['Unnamed: 0'])

# Drop column that all equal to NaN
data = data.dropna(axis=1, how='all')

# Fill NaN with "-"
data = data.fillna('-')

file_name2 = './Data/attributes_category1.txt'
with open(file_name2, 'r', encoding='utf-8') as f:
    attr_fix = f.read().splitlines()

regex_attr = ("|").join(attr_fix)
# clean the text and prepre some attribute
def clean_text_category1(msg):
    msg = str(msg).lower()
    msg = re.sub(r"smartphone[s]?","สมาร์ทโฟน",str(msg))
    msg = re.sub(r"computers และ laptops","คอมพิวเตอร์ และ แล็ปท็อป",str(msg))
    msg = re.sub(r"laptop[s]?","แล็ปท็อป",str(msg))
    msg = re.sub(r"desktops computers","คอมพิวเตอร์แบบตั้งโต๊ะ",str(msg))    
    
    if str(msg)=='โทรศัพท์มือถือและแท็บเล็ต':
        msg = re.sub(r"(โทรศัพท์มือถือและแท็บเล็ต)",r"หมวดหมู่ \1",str(msg))
    elif (str(msg)=='สมาร์ทโฟน') | (str(msg)=='แท็บเล็ต'):
        msg = re.sub(r"(สมาร์ทโฟน|แท็บเล็ต)",r"ประเภทย่อย \1",str(msg))
    elif str(msg)=='คอมพิวเตอร์ และ แล็ปท็อป':
        msg = re.sub(r"(คอมพิวเตอร์ และ แล็ปท็อป)",r"หมวดหมู่ \1",str(msg))
    elif (str(msg)=='แล็ปท็อป') | (str(msg)=='แล็ปท็อปสำหรับเล่นเกมส์'):
        msg = re.sub(r"(แล็ปท็อป|แล็ปท็อปสำหรับเล่นเกมส์)",r"ประเภทย่อย \1",str(msg))
    elif (str(msg)=='คอมพิวเตอร์แบบตั้งโต๊ะ') | (str(msg)=='gaming desktops'):
        msg = re.sub(r"(คอมพิวเตอร์แบบตั้งโต๊ะ|gaming desktops)",r"ประเภทย่อย \1",str(msg))
    elif re.match("[0-9]\*[0-9]",str(msg)):
        msg = re.sub(r"\*","x",str(msg))
    
    msg = re.sub(r"\&","และ",str(msg))
    msg = re.sub(r"sku","sku ",str(msg))
    msg = re.sub(r"\%","เปอร์เซนต์ ",str(msg))
    msg = re.sub(r"brand","แบรนด์ ",str(msg))
    msg = re.sub(r"model","โมเดล ",str(msg))
    msg = re.sub(r"กราฟฟิก|กราฟิก","graphics",str(msg))
    msg = re.sub(r"^color|^colour","สี ",str(msg))
    msg = re.sub(r"พาวเวอร์ซัพพลาย\s","power supply ",str(msg))
    msg = re.sub(r"เมนบอร์ด\s","mainboard ",str(msg))
    msg = re.sub(r"warranty\speriod\s+?","ระยะเวลาการรับประกัน ",str(msg))
    
    msg = re.sub(r"จอแสดงผล\s(size|screen)|[\W]จอแสดงผล\s|^screen\s|^monitor\s","หน้าจอแสดงผล ",str(msg))
    msg = re.sub(r"^resolution\s+?","ความละเอียด ",str(msg))
    msg = re.sub(r"^os\:?\s|operating\system\s+?","ระบบปฏิบัติการ ",str(msg))
    msg = re.sub(r"ความจุ\s*\(storage\)","ความจุที่เก็บข้อมูล ",str(msg))
    msg = re.sub(r"การ์ดจอ|การ์ดแสดงผล|graphics|graphic[s]? card|vga","gpu ",str(msg))
    msg = re.sub(r"ซีพียู\s","cpu ",str(msg))
    msg = re.sub(r"[\d+\s+]mp|เมกะพิกเซล\s?","ล้านพิกเซล ",str(msg))
    msg = re.sub(r"\s+แรม\s?|main\smemory\s|^memory\s+?[^r]"," ram ",str(msg))
    msg = re.sub(r"\s+รอม\s?"," rom ",str(msg))
    msg = re.sub(r"ความจุฮาร์ดไดร์ฟ|ฮาร์ดดิสก์\s?|hard\s?(disk|drive)\s?|hdd size"," hdd ",str(msg))
    msg = re.sub(r"display","จอแสดงผล ",str(msg))
    msg = re.sub(r"weight?","น้ำหนัก ",str(msg))
    msg = re.sub(r"[b]attery\s?[c]apacity","ความจุแบตเตอรี่ ",str(msg))
    msg = re.sub(r"แบต\s?^เตอรี่","แบตเตอรี่ ",str(msg))
    msg = re.sub(r"\w*\s?dimension w x d x h","ขนาด ",str(msg))
    msg = re.sub(r"\w*\s?dimension","ขนาด ",str(msg))
    msg = re.sub(r"พอร์ต|ports\s","port ",str(msg))
    
    msg = re.sub(r"^(amd ryzen|intel\w*i)",r"cpu \1",str(msg))
    msg = re.sub(r"^(nvidia|intel\w*graphic)",r"gpu \1",str(msg))
    msg = re.sub(r"^(\d+\.?\d?\sinch)",r"ขนาดหน้าจอ \1",str(msg))
    msg = re.sub(r"^(windows)",r"ระบบปฏิบัติการ \1",str(msg))
    msg = re.sub(r"\s(\d+\s)month[s]?",r"\1 เดือน",str(msg))
    msg = re.sub(r"\s(\d+\s)year[s]?",r"\1 ปี",str(msg))
    
    # หน่วย
    msg = re.sub(r"\฿,"," บาท ",str(msg))
    msg = re.sub(r"\sมิลลิเมตร|\sมม\.?|millimeter\.?"," mm ",str(msg))
    msg = re.sub(r"\sเซนติเมตร|\sซม\.?|centimeter\.?"," cm ",str(msg))
    msg = re.sub(r"\sกิโลกรัม|\sกก\.?|kilogram[s]?\.?"," kg ",str(msg))
    msg = re.sub(r"\sนิ้ว"," inch ",str(msg))
    
    # clean special character
    msg = re.sub(r"[\,]","",str(msg))
    msg = re.sub(r"[^a-z0-9ก-๙\.\s]"," ", str(msg))
    msg = re.sub(r"\s+"," ",str(msg)) 
    
    if any(ext in msg for ext in attr_fix):
        return msg
    else:
        return

# clean name
def clean_name(msg):
    msg = str(msg).lower()
    
    msg = re.sub(r"\&","และ",str(msg))
    msg = re.sub(r"sku","sku ",str(msg))
    msg = re.sub(r"brand","แบรนด์ ",str(msg))
    msg = re.sub(r"model","โมเดล ",str(msg))
    msg = re.sub(r"[\d\s]*มิลลิเมตร|\sมม\.?|millimeter\.?"," mm ",str(msg))
    msg = re.sub(r"[\d\s]*เซนติเมตร|\sซม\.?|centimeter\.?"," cm ",str(msg))
    msg = re.sub(r"\s+"," ",str(msg))
    
    return msg
    
# function for defining attribute of product
def define_attr(txt):
    p = re.compile('(หมวดหมู่|ประเภทย่อย|{}|)'.format(regex_attr), re.VERBOSE)
    #match = re.match(r"", str(txt))
    txt = re.sub(p,r"|\1|",str(txt))
    
    return txt

# Cleaning text with clean_text function and keep in dict
data_clean = {}
name_item = {}
for key,value in data.iteritems():
    tmp = []
    for i,txt in enumerate(value):
        if not txt == '-':
            if i==2 and re.match(r"category_[01]_\d+",str(key)):
                name_item[key] = clean_name(txt)
            elif i==3 and re.match(r"category_[2345678]_\d+",str(key)):
                name_item[key] = clean_name(txt)
            else:
                tmp.append(clean_text_category1(txt))
    
    tmp = [i for i in tmp if i] # Clear None value
    data_clean[key] = tmp 

# Mark attribute with '|'
attr_mark = {}
for key,value in data_clean.items():
    tmp = [define_attr(txt) for txt in value] 
    tmp = [re.sub(r"\|\|","",str(t)) for t in tmp]
    tmp = [i for i in tmp if re.search(r"\|",i)] 
    attr_mark[key] = tmp 
    
# Split attribute
def cap_attr(text):
    text = text.split("|")
    
    return text

# Split attribute
def cap_attr(text):
    text = text.split("|")
    
    return text

# Construct triples of product
triples = []
for i,v in attr.items():
    for l in v:
        idx=0
        for idx in range(len(l)-1):
            tmp_list = [i]
            if l[idx] in attr_fix:
                tmp_list.extend([l[idx],l[idx+1]])
                triples.append(tmp_list)
            idx += 1     
            
# Write the triples to text file
with open('./Data/triples_{}.txt'.format(file_name), 'w', encoding='utf-8') as f:
    for _list in triples:
        for _string in _list:
            #f.seek(0)
            f.write(str(_string) + '\t')
        f.write('\n')
        
# Product name and index
with open('./Data/Product_names_{}.txt'.format(file_name), 'w', encoding='utf-8') as fn:
    print(name_item, file=fn)
    
# Read triples from text file
with open ("Data/triples_01.txt", "r", encoding='utf-8') as myfile:
    triples = myfile.read().splitlines()
for i in range(len(triples)):
    triples[i] = triples[i].split('\t')
    triples[i] = [m for m in triples[i] if not m =='']
    
with open('./Data/Product_names.txt', 'r', encoding='utf-8') as fn: 
    content = fn.read() 
    dic = eval(content)
    
## Creating Nodes
nodes_e = []
nodes_c = []
relationships = []
for i in range(len(triples)):
    product = 0
    for n in nodes_e:
        if dict(n)['name'] == dic[triples[i][0]]:
            product = n
            break
    if product==0:
        product = Node('Product',name = dic[triples[i][0]])
        nodes_e.append(product)
    
    attr = 0
    for n in nodes_c:
        if dict(n)['name'] == triples[i][2]:
            attr = n 
            break
    if attr==0:
        attr = Node('Attribute',name = triples[i][2])
        nodes_c.append(attr)  
    
    relationship = Relationship(product,triples[i][1],attr)
    relationships.append(relationship)
    
# Creating the graph
graph =Graph(host='localhost',user='neo4j', password='5678') # initial graph from the data
#graph = Graph('bolt://neo4j:test@127.0.0.1:7687/db/data')
tx = graph.begin()

for node in nodes_e:
    tx.create(node)
for node in nodes_c:
    tx.create(node)
for relationship in relationships:
    tx.create(relationship)
    
tx.commit()

url = 'http://localhost:7474'
webbrowser.open(url, new=2) # new=2 opens a new tab




# In[ ]:




