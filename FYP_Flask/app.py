from flask import Flask,render_template,request
from model import model

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')
    
@app.route('/predict',methods=['POST'])

def predict():
    tweet = request.form['tweet']
    prediction = m.predict(tweet)
    return render_template('home.html',prediction= prediction)


if __name__ == "__main__":
    m = model()
    app.run(debug=True)