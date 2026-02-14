from flask import Flask, request, jsonify
from PIL import Image
import io
import os
import threading

from pycoral.adapters import common, detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

app = Flask(__name__)

MODEL_PATH = os.environ.get('CORAL_MODEL_PATH', '/app/models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite')
LABELS_PATH = os.environ.get('CORAL_LABELS_PATH', '/app/models/coco_labels.txt')

labels = {}
interpreter = None
in_w, in_h = 300, 300
init_error = None
init_lock = threading.Lock()


def init_engine():
    global labels, interpreter, in_w, in_h, init_error
    if interpreter is not None:
        return
    with init_lock:
        if interpreter is not None:
            return
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(MODEL_PATH)
            if not os.path.exists(LABELS_PATH):
                raise FileNotFoundError(LABELS_PATH)
            labels = read_label_file(LABELS_PATH)
            interpreter = make_interpreter(MODEL_PATH)
            interpreter.allocate_tensors()
            in_w, in_h = common.input_size(interpreter)
            init_error = None
        except Exception as e:
            init_error = str(e)
            interpreter = None


@app.route('/health', methods=['GET'])
def health():
    ready = interpreter is not None
    return jsonify({
        'ok': True,
        'ready': ready,
        'model': os.path.basename(MODEL_PATH),
        'init_error': init_error,
    })


@app.route('/infer', methods=['POST'])
def infer():
    init_engine()
    if interpreter is None:
        return jsonify({'error': f'coral init failed: {init_error}'}), 503

    data = request.get_data()
    if not data:
        return jsonify({'error': 'empty body'}), 400

    try:
        img = Image.open(io.BytesIO(data)).convert('RGB').resize((in_w, in_h))
    except Exception as e:
        return jsonify({'error': f'invalid image: {e}'}), 400

    common.set_input(interpreter, img)
    interpreter.invoke()

    threshold = float(request.args.get('threshold', '0.35'))
    objs = detect.get_objects(interpreter, score_threshold=threshold)

    out = []
    for o in objs:
        out.append({
            'id': int(o.id),
            'label': labels.get(int(o.id), str(o.id)),
            'score': float(o.score),
            'bbox': {
                'xmin': float(o.bbox.xmin),
                'ymin': float(o.bbox.ymin),
                'xmax': float(o.bbox.xmax),
                'ymax': float(o.bbox.ymax),
            }
        })

    return jsonify({'detections': out, 'input_size': [int(in_w), int(in_h)]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8765)
