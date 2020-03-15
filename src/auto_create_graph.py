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

# clean the text and prepre some attribute
def clean_text_category1(msg, attr_fix):
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
    elif (str(msg)=='คอมพิวเตอร์แบบตั้งโต๊ะ') | (str(msg)=='gaming desktops') | (str(msg)=='คอมประกอบ'):
        msg = re.sub(r"(คอมประกอบ|คอมพิวเตอร์แบบตั้งโต๊ะ|gaming desktops)",r"ประเภทย่อย \1",str(msg))
    elif re.match(r"[0-9]\*[0-9]",str(msg)):
        msg = re.sub(r"\*","x",str(msg))
    
    msg = re.sub(r"\&","และ",str(msg))
    msg = re.sub(r"sku","sku ",str(msg))
    msg = re.sub(r"\%","เปอร์เซนต์ ",str(msg))
    msg = re.sub(r"brand","แบรนด์ ",str(msg))
    msg = re.sub(r"model","โมเดล ",str(msg))
    msg = re.sub(r"กราฟฟิก|กราฟิก","graphics",str(msg))
    msg = re.sub(r"^color|^colour","สี ",str(msg))
    msg = re.sub(r"พาวเวอร์ซัพพลาย\s|power\ssupply\s","psu ",str(msg))
    msg = re.sub(r"เมนบอร์ด\s|^mb\s","mainboard ",str(msg))
    msg = re.sub(r"warranty\speriod\s+?","ระยะเวลาการรับประกัน ",str(msg))
    msg = re.sub(r"\sprocessor\s","cpu ",str(msg))
    msg = re.sub(r"เลโนโว่"," lenovo ",str(msg))
    msg = re.sub(r"ซัมซุง"," samsung ",str(msg))
    
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
    
    # หน่วย
    msg = re.sub(r"\฿,"," บาท ",str(msg))
    msg = re.sub(r"\sมิลลิเมตร|\sมม\.?|millimeter\.?"," mm ",str(msg))
    msg = re.sub(r"\sเซนติเมตร|\sซม\.?|centimeter\.?"," cm ",str(msg))
    msg = re.sub(r"\sกิโลกรัม|\sกก\.?|kilogram[s]?\.?"," kg ",str(msg))
    msg = re.sub(r"\sนิ้ว"," inch ",str(msg))
    
    msg = re.sub(r"^(amd ryzen|intel\w*[ip])",r"cpu \1",str(msg))
    msg = re.sub(r"^(nvidia|intel\w*graphic)",r"gpu \1",str(msg))
    msg = re.sub(r"^(\d+\.?\d?\sinch)",r"ขนาดหน้าจอ \1",str(msg))
    msg = re.sub(r"^(windows)",r"ระบบปฏิบัติการ \1",str(msg))
    msg = re.sub(r"\s(\d+\s)month[s]?",r"\1 เดือน",str(msg))
    msg = re.sub(r"\s(\d+\s)year[s]?",r"\1 ปี",str(msg))
    
    
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

    # clean special character
    msg = re.sub(r"[\,]","",str(msg))
    msg = re.sub(r"【.*】","", str(msg))
    msg = re.sub(r"[^a-z0-9ก-๙\.\s]"," ", str(msg))
    msg = re.sub(r"\s+"," ",str(msg))
    
    return msg
    
# function for defining attribute of product
def define_attr(txt,regex_attr):
    p = re.compile('(หมวดหมู่|ประเภทย่อย|{}|)'.format(regex_attr), re.VERBOSE)
    #match = re.match(r"", str(txt))
    txt = re.sub(p,r"|\1|",str(txt))
    
    return txt

# Split attribute
def cap_attr(text):
    text = text.split("|")
    
    return text



def create_graph(triples,dic,host,user,password):
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
    graph =Graph(host=host,user=user, password=password) # initial graph from the data
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
    return




# In[ ]:




