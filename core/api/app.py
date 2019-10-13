"""Flask web server serving text_recognizer predictions."""
# From https://github.com/UnitedIncome/serverless-python-requirements
try:
    import unzip_requirements  # pylint: disable=unused-import
except ImportError:
    pass

from flask import Flask, request, jsonify

from text_recognizer.line_predictor import LinePredictor
from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizer
import text_recognizer.util as util

app = Flask(__name__)

# init predictor
predictor = LinePredictor()
text_ocr = ParagraphTextRecognizer()

@app.route('/')
def index():
    return 'Hello, world!'


@app.route('/v1/predict', methods=['GET', 'POST'])
def predict():
    image = _load_image()
    pred, conf = predictor.predict(image)
    print("METRIC confidence {}".format(conf))
    print("METRIC mean_intensity {}".format(image.mean()))
    print("INFO pred {}".format(pred))
    return jsonify({'pred': str(pred), 'conf': float(conf)})

@app.route('/v1/ocr', methods=['GET', 'POST'])
def ocr():
    image = _load_image()
    pred, crops = text_ocr.predict(image)
    print("INFO pred {}".format(pred))
    return jsonify({'pred': str(pred)})

def _load_image():
    if request.method == 'POST':
        data = request.get_json()
        if data is None:
            return 'no json received'
        return util.read_b64_image(data['image'], grayscale=True)
    if request.method == 'GET':
        image_url = request.args.get('image_url')
        if image_url is None:
            return 'no image_url defined in query string'
        print("INFO url {}".format(image_url))
        return util.read_image(image_url, grayscale=True)
    raise ValueError('Unsupported HTTP method')


def main():
    app.run(host='0.0.0.0', port=8000, debug=False)  # nosec


if __name__ == '__main__':
    main()
