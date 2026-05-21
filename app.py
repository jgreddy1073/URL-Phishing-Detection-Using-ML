from flask import Flask, request, render_template, redirect
from featureExtractor import featureExtraction
from pycaret.classification import load_model, predict_model

model = load_model('model/phishingdetection')

def predict(url):
    data = featureExtraction(url)
    result = predict_model(model, data=data)
    prediction_score = result['prediction_score'][0]  
    prediction_label = result['prediction_label'][0] 
    
    return {
        'prediction_label': prediction_label,
        'prediction_score': prediction_score * 100,
    }

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    data = None
    url = None
    
    if request.method == "POST":
        url = request.form["url"]
        data = predict(url)
        return render_template('index.html', url=url, data=data)
    return render_template("home.html", data=data)
@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/index")
def in_htm():
    return render_template("index.html")

@app.route("/logout")
def logout():
    return render_template("home.html")
@app.route("/home")
def home():
    return render_template("home.html")
@app.route("/faqs")
def faqs():
    return render_template("faqs.html")
@app.route("/contact")
def contact():
    return render_template("contact.html")
@app.route("/open_url")
def open_url():
    url = request.args.get("url")
    return redirect(url) if url else redirect("/")

if __name__ == "__main__":
    app.run(debug=True)