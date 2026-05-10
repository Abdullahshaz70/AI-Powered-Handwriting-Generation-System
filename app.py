"""
Flask web interface for the Handwriting Generation System.

Run: python app.py  →  http://localhost:5000
"""
import os, sys, io, base64, traceback
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from generate import load_model, generate_word, export_pdf
from data     import CHAR_TO_IDX

app = Flask(__name__)

_MODEL = _CKPT = _DEVICE = _REFS = None
OUTPUTS = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUTPUTS, exist_ok=True)


def _get_model():
    global _MODEL, _CKPT, _DEVICE, _REFS
    if _MODEL is None:
        _MODEL, _CKPT, _DEVICE, _REFS = load_model()
    return _MODEL, _CKPT, _DEVICE, _REFS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/status')
def status():
    try:
        _, ckpt, _, _ = _get_model()
        return jsonify({
            'ready':   True,
            'epoch':   ckpt.get('epoch'),
            'loss':    round(float(ckpt.get('val_loss', 0)), 6),
            'writers': ckpt.get('writer_names', []),
        })
    except Exception as e:
        return jsonify({'ready': False, 'error': str(e)})


@app.route('/generate', methods=['POST'])
def generate():
    data        = request.get_json(force=True)
    text        = data.get('text', '').strip()
    writer_idx  = int(data.get('writer_idx', 0))
    noise_scale = float(data.get('noise', 0.018))

    supported = ''.join(c for c in text if c in CHAR_TO_IDX)
    if not supported:
        return jsonify({'error': 'Input must contain A-Z, a-z, or 0-9'}), 400

    try:
        model, ckpt, device, refs = _get_model()
        num_writers = ckpt.get('num_writers', 6)
        writer_idx  = max(0, min(writer_idx, num_writers - 1))

        _, page = generate_word(model, supported, writer_idx=writer_idx,
                                device=device, refs=refs, noise_scale=noise_scale)

        buf = io.BytesIO()
        Image.fromarray(page).save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode()
        return jsonify({'image': b64, 'rendered': supported})

    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'Generation failed — see server log'}), 500


@app.route('/download', methods=['POST'])
def download():
    data        = request.get_json(force=True)
    text        = data.get('text', '').strip()
    writer_idx  = int(data.get('writer_idx', 0))
    noise_scale = float(data.get('noise', 0.018))

    supported = ''.join(c for c in text if c in CHAR_TO_IDX)
    if not supported:
        return jsonify({'error': 'No supported characters'}), 400

    try:
        model, ckpt, device, refs = _get_model()
        num_writers = ckpt.get('num_writers', 6)
        writer_idx  = max(0, min(writer_idx, num_writers - 1))

        _, page = generate_word(model, supported, writer_idx=writer_idx,
                                device=device, refs=refs, noise_scale=noise_scale)
        pdf_path = os.path.join(OUTPUTS, 'handwriting.pdf')
        export_pdf(page, pdf_path)
        return send_file(pdf_path, as_attachment=True, download_name='handwriting.pdf')

    except Exception:
        traceback.print_exc()
        return jsonify({'error': 'PDF export failed'}), 500


if __name__ == '__main__':
    print('Loading model...')
    _get_model()
    print('Ready → http://localhost:5000')
    app.run(debug=False, host='0.0.0.0', port=5000)
