from flask import Flask

UPLOAD_FOLDER = 'C:/Users/super/Documents/GitHub/ProductGraph/src/files/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER