# -*- coding: utf-8 -*-
from pythainlp import word_tokenize
from pythainlp.corpus import thai_stopwords
from pythainlp.corpus import wordnet
from pythainlp.ulmfit import *
from nltk.stem.porter import PorterStemmer
from nltk.corpus import words
from stop_words import get_stop_words
from fastai.text import *

import re
import string
import nltk
nltk.download('words')
th_stop = tuple(thai_stopwords())
en_stop = tuple(get_stop_words('en'))
p_stemmer = PorterStemmer()

def clean_msg(msg):
    
    # ลบ text ที่อยู่ในวงเล็บ <> 
    msg = re.sub(r'<.*#?>','', msg)
    # ลบ hashtag
    msg = re.sub(r'#','',msg)
    
    # ลบ เครื่องหมายคำพูด 
    for c in string.punctuation:
        msg = re.sub(r'\{}'.format(c),'',msg)
    
    # ลบ separator เช่น \n \t
    msg = ' '.join(msg.split())
    
    return msg

def split_word(text):
    words = word_tokenize(text,engine='newmm')

    # Remove stop words TH and EN
    words = [i for i in words if not i in th_stop and not i in en_stop]

    # รากศัพท์
    # EN
    words = [p_stemmer.stem(i) for i in words]
    
    # TH
    tokens_temp=[]
    for i in words:
        w_syn = wordnet.synsets(i)
        if (len(w_syn)>0) and (len(w_syn[0].lemma_names('tha'))>0):
            tokens_temp.append(w_syn[0].lemma_names('tha')[0])
        else:
            tokens_temp.append(i)
    
    words = tokens_temp
    
    # ลบช่องว่าง
    words = [i for i in words if not ' ' in i]

    return words


text_list = ["#อีกหนึ่งในความทันสมัยเพิ่มขึ้นด้วยที่กรองใบชา และเศษเครื่องดื่ม ที่อยู่ส่วนบนปากของกา ตัวเครื่องผลิตจากสแตนเลส ไม่เป็นสนิม ทำให้น้ำร้อนได้เร็วทันใจฝาเปิดค้างไว้ได้เพียงกดปุ่ม และกดฝาให้ลงล็อคเมื่อปิดฝา ตัวฐานหมุนรอบได้ 360 องศา ยกตัวกาออกจากฐานได้จึงทำให้สะดวก เหมาะสำหรับการพกพาไปใช้ในสถานที่ต่าง ๆ อีกทั้งยังมีขีดบอกระดับน้ำที่ดูได้ง่าย มีไฟแสดงสถานะความร้อน พร้อมระบบตัดไฟฟ้าอัตโนมัติเมื่อน้ำเดือดแล้ว หรือมีความร้อนสูง จึงมีความปลอดภัยยิ่งขึ้น", "คุณสมบัติต้มน้ำร้อนได้เร็ว ประมาณ 4 นาที ตัวเครื่องทำจากสแตนเลส ไม่เป็นสนิมฮีตเตอร์แบบขดลวด ทำจากสแตนเลสระบบตัดไฟอัตโนมัติเมื่อน้ำเดือด หรือน้ำแห้งมีที่กรองเศษใบชา และกากเครื่องดื่ม ตรงส่วนปากของกาเหมาะอย่างยิ่งสำหรับใช้ในห้องพักโรงแรม หรือพกพาท่องเที่ยวตัวกาสามารถแยกออกจากฐานได้ตัวฐานหมุนได้รอบ 360 องศาความจุ 1.8 ลิตรขนาด กว้าง 17 ซม. (วัดจากใต้ฐาน) * สูง 23 ซม.(รวมฐาน)/ สูง 20.5 ซม.(ไม่รวมฐาน)กำลังไฟ 1500w 220v 50Hz"]
text_clean = []
text_clean = [clean_msg(text) for text in text_list]
print(text_clean)

vect = document_vector(text_clean[0],)
print(vect)

words = [split_word(text) for text in text_clean]
print(words)




