import os
from flask import Flask
from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from flask import Flask, render_template, redirect, url_for, jsonify, abort, request
from flask_cors import CORS, cross_origin
import uuid

app = Flask(__name__, static_url_path='/static')
cors = CORS(app)


class Register(FlaskForm):
    file = FileField(label='file')


@app.route('/hello')
@app.route('/hello/<name>')
def hello(name=None):

    return render_template('hello.html',  name=name)


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get("file")
    filename = str(uuid.uuid4()) + "_" + file.filename
    filepath = os.path.join("static", filename)
    file.save(filepath)
    url = "http://localhost:3101/static/"+filename
    rep = '{"status":"200","url":"%s","info":"0,0,0,0"}'
    return rep%(url)


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=3101,  debug=True)
