import os
#import magic
import urllib.request
from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename

import pandas as pd
import io
import csv
import re
import string
import nltk
from auto_create_graph import *

from nltk.corpus import words
from stop_words import get_stop_words

import webbrowser
from py2neo import Node, Relationship, Graph

ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('uploads.html')

@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			flash('No file')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			#file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

			# input
			df = pd.read_csv(file)

			file_name2 = os.path.join(app.root_path, 'data/attributes_category1.txt')

		
			with open(file_name2, 'r', encoding='utf-8') as f:
				attr_fix = f.read().splitlines()

			# Drop a column that not use
			data = df.drop(columns=['Unnamed: 0'])
			# Drop column that all equal to NaN
			data = data.dropna(axis=1, how='all')
			# Fill NaN with "-"
			data = data.fillna('-')
			# Filter
			data = data.filter(regex ='category_[0-9]_[0-9]+')
			regex_attr = ("|").join(attr_fix)

			# Cleaning text with clean_text function and keep in dict
			data_clean = {}
			name_item = {}
			for key,value in data.iteritems():
				tmp = []
				for i,txt in enumerate(value):
					if not txt == '-':
						if i==2 and re.match(r"category_[01]_\d+",str(key)):
							name_item[key] = clean_name(txt)
						elif i==3 and re.match(r"category_[23456789]_\d+",str(key)):
							name_item[key] = clean_name(txt)
						else:
							tmp.append(clean_text_category1(txt,attr_fix))
				tmp = [i for i in tmp if i] # Clear None value
				data_clean[key] = tmp

			# Mark attribute with '|'
			attr_mark = {}
			for key,value in data_clean.items():
				tmp = [define_attr(txt,regex_attr) for txt in value]
				tmp = [re.sub(r"\|\|","",str(t)) for t in tmp]
				tmp = [i for i in tmp if re.search(r"\|",i)]
				attr_mark[key] = tmp 

			# capture attribute by split the marker
			attr = {}
			for i,v in attr_mark.items():
				tmp_list = []
				for txt in v:
					match = cap_attr(txt)
					match = [m for m in match if not m =='' and not m == ' ']
					tmp_list.append(match)
				attr[i] = tmp_list

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
			with open('./data/triples_{}.txt'.format(filename), 'w', encoding='utf-8') as f:
				for _list in triples:
					for _string in _list:
						#f.seek(0)
						f.write(str(_string) + '\t')
						f.write('\n')

			# Write product name and index
			with open('./data/Product_names_{}.txt'.format(filename), 'w', encoding='utf-8') as fn:
				print(name_item, file=fn)

			create_graph(triples,name_item,'localhost','neo4j','1234')

			flash('Completed!')
			return redirect('/')
		else:
			flash('Allowed file type is csv')
			return redirect(request.url)

if __name__ == "__main__":
	app.run(debug=True)