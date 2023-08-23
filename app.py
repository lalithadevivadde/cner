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


def ff(x, score):
    if len(x) == 0:
        return ""
    return f'{x[0]} ({np.round(score[0], 2) * 100}%)'


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
            if score > 0.01:
                if (label == "VENDOR") & (len(vendor) == 0):
                    vendor.append(doc[start:end])
                    vendor_score.append(score)
                elif (label == "CLIENT") & (len(client) == 0):
                    client.append(doc[start:end])
                    client_score.append(score)
                elif (label == "VALUE") & (len(value) == 0):
                    value.append(doc[start:end])
                    value_score.append(score)
                elif (label == "DURATION") & (len(duration) == 0):
                    duration.append(doc[start:end])
                    duration_score.append(score)
                elif (label == "LOCATION") & (len(location) == 0):
                    location.append(doc[start:end])
                    location_score.append(score)
                if (label == "PRODUCT") & (len(product) == 0):
                    product.append(doc[start:end])
                    product_score.append(score)

        result = f'<b>VENDOR:</b> {ff(vendor, vendor_score)}<br><b>CLIENT:</b> {ff(client, client_score)}<br><b>VALUE:</b> ' \
                 f'{ff(value, value_score)}<br><b>DURATION:</b> {ff(duration, duration_score)}' \
                 f'<br><b>LOCATION:</b> {ff(location, location_score)}<br><b>PRODUCT:</b> {ff(product, product_score)}'


        result = f'<h3>Recognized Entities</h3><p style="color: black;">{result}</p>'
    except Exception as e:
        result = f'Error: {e}'
    return render_template("index.html", result=result, input=texts[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
