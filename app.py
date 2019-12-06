# _*_ coding: utf-8 _*_
"""
@author: Jibao Wang
@time: 2019/11/29 15:43
"""
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import execute


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/message", methods=['POST'])
def reply():
    sentence = request.form['msg']
    translate, _ = execute.translate(sentence)
    translate = ' '.join(translate.strip().split(' ')[:-1])
    print(sentence)
    print(translate)
    return jsonify(text=translate)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=12345)