"""Local JSON API for the exported churn pipeline. Run from the repository root."""
import argparse
import json
import math
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_artifact(path):
    artifact = joblib.load(path)
    if artifact.get('artifact_version') != 1:
        raise ValueError('Unsupported model artifact version.')
    return artifact


def predict_customer(artifact, customer):
    if not isinstance(customer, dict):
        raise ValueError('customer must be a JSON object.')
    fields = artifact['fields']
    expected = {field['name'] for field in fields}
    missing = expected - customer.keys()
    extra = customer.keys() - expected
    if missing or extra:
        raise ValueError(f'Missing fields: {sorted(missing)}. Unexpected fields: {sorted(extra)}.')
    for field in fields:
        value = customer[field['name']]
        if value is None:
            continue  # The exported pipeline handles missing values.
        if field['type'] == 'number':
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field['name']} must be a finite number or null.")
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field['name']} must be a nonempty string or null.")
    record = {field['name']: customer[field['name']] if customer[field['name']] is not None
              else float('nan') for field in fields}
    frame = pd.DataFrame([record], columns=[field['name'] for field in fields])
    model = artifact['pipeline']
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Found unknown categories')
        prediction = int(model.predict(frame)[0])
        probability = float(model.predict_proba(frame)[0, list(model.classes_).index(1)])
    return {'prediction': prediction, 'label': 'Churn' if prediction else 'No churn',
            'churn_probability': probability, 'model': artifact['model_name']}


def make_handler(artifact):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == '/health':
                return self.reply(200, {'status': 'ok', 'model': artifact['model_name']})
            if self.path == '/schema':
                return self.reply(200, {'fields': artifact['fields'],
                                        'example': {'customer': artifact['example']},
                                        'positive_class': 'Churn', 'versions': artifact['versions']})
            return self.reply(404, {'error': 'Use GET /health, GET /schema or POST /predict.'})

        def do_POST(self):
            if self.path != '/predict':
                return self.reply(404, {'error': 'Endpoint not found.'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if size <= 0 or size > 65536:
                    raise ValueError('Provide a JSON request body of at most 64 KB.')
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict) or set(payload) != {'customer'}:
                    raise ValueError('Request must contain exactly one customer object.')
                result = predict_customer(artifact, payload['customer'])
            except (ValueError, UnicodeDecodeError) as exc:
                return self.reply(400, {'error': str(exc)})
            except Exception:
                return self.reply(500, {'error': 'Prediction failed; check server logs and model compatibility.'})
            return self.reply(200, result)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--model', type=Path, default=ROOT / 'models/churn_pipeline.joblib')
    args = parser.parse_args()
    artifact = load_artifact(args.model)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(artifact))
    print(f'Churn API running at http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
