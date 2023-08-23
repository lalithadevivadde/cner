from flask import Flask, render_template, request
from flask_cors import cross_origin, CORS
import spacy
import os


app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)


@app.route("/")
@cross_origin()
def index():
    """
    displays the index.html page
    """
    return render_template("index.html")


def ff(x):
    if len(x) == 0:
        return ""
    return x[0]

@app.route("/", methods=["POST"])
@cross_origin()
def text_summarization():
    try:
        text = request.form["text_input"]
        nlp = spacy.load(os.path.join(".", "ner_model"))
        doc = nlp(text)
        vendor = []
        client = []
        value = []
        duration = []
        product = []
        for i in doc.ents:
            if i.label_ == 'VENDOR':
                vendor.append(i.text)
            elif i.label_ == 'CLIENT':
                client.append(i.text)
            elif i.label_ == 'VALUE':
                value.append(i.text)
            elif i.label_ == 'DURATION':
                duration.append(i.text)
            else:
                product.append(i.text)

        result = f'<b>VENDOR:</b> {ff(vendor)}<br><b>CLIENT:</b> {ff(client)}<br><b>VALUE:</b> ' \
                 f'{ff(value)}<br><b>DURATION:</b> {ff(duration)}<br><b>PRODUCT:</b> {ff(product)}'

        result = f'<h3>Recognized Entities</h3><p style="color: black;">{result}</p>'
    except Exception as e:
        result = f'Error: {e}'
    return render_template("index.html", result=result, input=text)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
