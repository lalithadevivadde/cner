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
        location = []
        for i in doc.ents:
            if (i.label_ == 'VENDOR') & (len(vendor) == 0):
                vendor.append(i.text)
            elif (i.label_ == 'CLIENT') & (len(client) == 0):
                client.append(i.text)
            elif (i.label_ == 'VALUE') & (len(value) == 0):
                value.append(i.text)
            elif (i.label_ == 'DURATION') & (len(duration) == 0):
                duration.append(i.text)
            elif (i.label_ == 'PRODUCT') & (len(product) == 0):
                product.append(i.text)
            elif (i.label_ == 'LOCATION') & (len(location) == 0):
                location.append(i.text)

        result = f'<b>VENDOR:</b> {ff(vendor)}<br><b>CLIENT:</b> {ff(client)}<br><b>VALUE:</b> ' \
                 f'{ff(value)}<br><b>DURATION:</b> {ff(duration)}' \
                 f'<br><b>LOCATION:</b> {ff(location)}<br><b>PRODUCT:</b> {ff(product)}'

        result = f'<h3>Recognized Entities</h3><p style="color: black;">{result}</p>'
    except Exception as e:
        result = f'Error: {e}'
    return render_template("index.html", result=result, input=text)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
