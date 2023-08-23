from flask import Flask, render_template, request
from flask_cors import cross_origin, CORS
import spacy
import numpy as np
import os
from collections import defaultdict

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)


@app.route("/")
@cross_origin()
def index():
    """
    displays the index.html page
    """
    return render_template("index.html")


@app.route("/", methods=["POST"])
@cross_origin()
def text_summarization():
    try:
        texts = [request.form["text_input"]]

        beam_width = 16
        beam_density = 0.0001
        nlp = spacy.load(os.path.join(".", "ner_model"))
        er = nlp.get_pipe("ner")

        with nlp.disable_pipes('ner'):
            docs = list(nlp.pipe(texts))
        beams = er.beam_parse(docs, beam_width=beam_width, beam_density=beam_density)

        for doc, beam in zip(docs, beams):
            entity_scores = defaultdict(float)
            for score, ents in er.moves.get_beam_parses(beam):
                for start, end, label in ents:
                    entity_scores[(start, end, label)] += score

        vendor = []
        client = []
        value = []
        duration = []
        location = []
        product = []

        vendor_score = []
        client_score = []
        value_score = []
        duration_score = []
        location_score = []
        product_score = []

        for key in entity_scores:
            start, end, label = key
            score = entity_scores[key]
            if (score > 0.01):
                if label == "VENDOR":
                    vendor.append(doc[start:end])
                    vendor_score.append(score)
                elif label == "CLIENT":
                    client.append(doc[start:end])
                    client_score.append(score)
                elif label == "VALUE":
                    value.append(doc[start:end])
                    value_score.append(score)
                elif label == "DURATION":
                    duration.append(doc[start:end])
                    duration_score.append(score)
                elif label == "LOCATION":
                    location.append(doc[start:end])
                    location_score.append(score)
                if label == "PRODUCT":
                    product.append(doc[start:end])
                    product_score.append(score)

        if len(vendor) > 0:
            vendor_argmax = np.argmax(vendor_score)
            vendor = f'{vendor[vendor_argmax]} ({np.round(vendor_score[vendor_argmax], 2) * 100}%)'
        else:
            vendor = ""

        if len(client) > 0:
            client_argmax = np.argmax(client_score)
            client = f'{client[client_argmax]} ({np.round(client_score[client_argmax], 2) * 100}%)'
        else:
            client = ""

        if len(value) > 0:
            value_argmax = np.argmax(value_score)
            value = f'{value[value_argmax]} ({np.round(value_score[value_argmax], 2) * 100}%)'
        else:
            value = ""

        if len(duration) > 0:
            duration_argmax = np.argmax(duration_score)
            duration = f'{duration[duration_argmax]} ({np.round(duration_score[duration_argmax], 2) * 100}%)'
        else:
            duration = ""

        if len(location) > 0:
            location_argmax = np.argmax(location_score)
            location = f'{location[location_argmax]} ({np.round(location_score[location_argmax], 2) * 100}%)'
        else:
            location = ""

        if len(product) > 0:
            product_argmax = np.argmax(product_score)
            product = f'{product[product_argmax]} ({np.round(product_score[product_argmax], 2) * 100}%)'
        else:
            product = ""

        print(vendor)

        result = f'<b>VENDOR:</b> {vendor}<br><b>CLIENT:</b> {client}<br><b>VALUE:</b> ' \
                 f'{value}<br><b>DURATION:</b> {duration}' \
                 f'<br><b>LOCATION:</b> {location}<br><b>PRODUCT:</b> {product}'

        result = f'<h3>Recognized Entities</h3><p style="color: black;">{result}</p>'
    except Exception as e:
        result = f'Error: {e}'
    return render_template("index.html", result=result, input=texts[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
