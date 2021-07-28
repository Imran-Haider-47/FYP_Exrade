from logging import Handler
from scrapper import scrapper
from flask import Flask,render_template,request
from model import model
from scrapper import scrapper
import json
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])


def predict():
    tweet = request.form['tweet']
    prediction = m.predict(tweet)
    return render_template('home.html',prediction=prediction)


    
@app.route('/profile',methods=['POST','GET'])
def profile():
    if request.method=='GET':
        return render_template('menu.html')

    link = request.form['link']
    tweets_array=scrap.get_tweets(link)
    result=[]
    for tweet in tweets_array:
        label=m.predict(tweet)
        label=(tweet,*label)
        result.append(label)
    print(result)
    return render_template('menu.html',result=result)

if __name__ == "__main__":
    m = model()
    scrap= scrapper()
    app.run(debug=True)