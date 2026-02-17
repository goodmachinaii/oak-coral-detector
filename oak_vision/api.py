from __future__ import annotations
import json
import threading
from pathlib import Path


def create_app(storage, state: dict):
    from flask import Flask, jsonify, request, send_from_directory

    app = Flask(__name__, static_folder=str((Path(__file__).resolve().parent.parent / 'front').resolve()))

    def _int(name: str, default: int):
        raw = request.args.get(name, str(default))
        try:
            return int(raw), None
        except ValueError:
            return None, jsonify({'error': f'invalid {name} parameter'}), 400

    @app.get('/status')
    def status():
        return jsonify({
            'mode': state.get('mode', 'unknown'),
            'fps': round(float(state.get('fps', 0.0)), 2),
            'infer_ms': round(float(state.get('infer_ms', 0.0)), 2),
            'running': bool(state.get('running', True)),
        })

    @app.get('/latest')
    def latest():
        limit = int(request.args.get('limit', '20'))
        return jsonify({'objects': storage.get_latest(limit=limit)})

    @app.get('/events')
    def events():
        try:
            hours = int(request.args.get('hours', '24'))
        except ValueError:
            return jsonify({'error': 'invalid hours parameter'}), 400
        return jsonify(storage.get_events(hours=hours, label=request.args.get('label'), status=request.args.get('status')))

    @app.get('/events/<int:event_id>')
    def event_detail(event_id: int):
        ev = storage.get_event_detail(event_id)
        if not ev:
            return jsonify({'error': 'event not found'}), 404
        return jsonify(ev)

    @app.get('/stats')
    def stats():
        try:
            hours = int(request.args.get('hours', '24'))
        except ValueError:
            return jsonify({'error': 'invalid hours parameter'}), 400
        return jsonify(storage.get_stats(hours=hours))

    @app.get('/health')
    def health():
        return jsonify({
            'oak_connected': bool(state.get('oak_connected', True)),
            'coral_mode': state.get('mode', 'unknown'),
            'uptime_sec': int(state.get('uptime_sec', 0)),
            'fps': round(float(state.get('fps', 0.0)), 2),
            'events_active': sum((storage.get_stats(hours=24).get('active_now') or {}).values()),
            'last_errors': storage.get_last_errors(limit=8),
            'db_size_mb': storage.get_db_size_mb(),
        })

    @app.get('/')
    def root():
        return send_from_directory(app.static_folder, 'index.html')

    @app.get('/front/<path:name>')
    def front_files(name):
        return send_from_directory(app.static_folder, name)

    return app


class ApiServer:
    def __init__(self, storage, state: dict, host='0.0.0.0', port=5000):
        self.storage = storage
        self.state = state
        self.host = host
        self.port = int(port)
        self.thread = None
        self.server = None

    def _start_stdlib_fallback(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from urllib.parse import urlparse, parse_qs

        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, obj, code=200, content_type='application/json'):
                data = json.dumps(obj).encode('utf-8') if content_type == 'application/json' else obj
                self.send_response(code)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _int(self, qs, name, default):
                raw = (qs.get(name) or [str(default)])[0]
                try:
                    return int(raw), None
                except ValueError:
                    return None, {'error': f'invalid {name} parameter'}

            def do_GET(self):
                p = urlparse(self.path)
                qs = parse_qs(p.query)

                if p.path == '/status':
                    return self._send({
                        'mode': outer.state.get('mode', 'unknown'),
                        'fps': round(float(outer.state.get('fps', 0.0)), 2),
                        'infer_ms': round(float(outer.state.get('infer_ms', 0.0)), 2),
                        'running': bool(outer.state.get('running', True)),
                    })

                if p.path == '/latest':
                    limit, err = self._int(qs, 'limit', 20)
                    if err:
                        return self._send(err, 400)
                    return self._send({'objects': outer.storage.get_latest(limit=limit)})

                if p.path == '/events':
                    hours, err = self._int(qs, 'hours', 24)
                    if err:
                        return self._send(err, 400)
                    label = (qs.get('label') or [None])[0]
                    status = (qs.get('status') or [None])[0]
                    return self._send(outer.storage.get_events(hours=hours, label=label, status=status))

                if p.path.startswith('/events/'):
                    try:
                        eid = int(p.path.split('/')[-1])
                    except Exception:
                        return self._send({'error': 'event not found'}, 404)
                    ev = outer.storage.get_event_detail(eid)
                    if not ev:
                        return self._send({'error': 'event not found'}, 404)
                    return self._send(ev)

                if p.path == '/stats':
                    hours, err = self._int(qs, 'hours', 24)
                    if err:
                        return self._send(err, 400)
                    return self._send(outer.storage.get_stats(hours=hours))

                if p.path == '/health':
                    stats = outer.storage.get_stats(hours=24)
                    return self._send({
                        'oak_connected': bool(outer.state.get('oak_connected', True)),
                        'coral_mode': outer.state.get('mode', 'unknown'),
                        'uptime_sec': int(outer.state.get('uptime_sec', 0)),
                        'fps': round(float(outer.state.get('fps', 0.0)), 2),
                        'events_active': sum((stats.get('active_now') or {}).values()),
                        'last_errors': outer.storage.get_last_errors(limit=8),
                        'db_size_mb': outer.storage.get_db_size_mb(),
                    })

                if p.path in ('/', '/front/index.html'):
                    index = (Path(__file__).resolve().parent.parent / 'front' / 'index.html')
                    if index.exists():
                        return self._send(index.read_bytes(), content_type='text/html; charset=utf-8')
                return self._send({'error': 'not found'}, 404)

            def log_message(self, format, *args):
                return

        self.server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return True, 'stdlib-http'

    def start(self):
        try:
            from werkzeug.serving import make_server
            app = create_app(self.storage, self.state)
            self.server = make_server(self.host, self.port, app)
            self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.thread.start()
            return True, 'flask'
        except Exception:
            return self._start_stdlib_fallback()

    def stop(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
