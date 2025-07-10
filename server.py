from http.server import BaseHTTPRequestHandler, HTTPServer
import json


class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        response_code = 200
        response = ''

        try:
            content_length = int(self.headers.get('Content-Length'))
            content = self.rfile.read(content_length)
            payload = json.loads(content)

            if payload.get('train'):
                nn.train(payload['trainArray'])
                nn.save()

            elif payload.get('predict'):
                try:
                    response = {
                        'type': 'test',
                        'result': nn.predict(str(payload['image']))
                    }
                except:
                    response_code = 500

            else:
                response_code = 400

        except Exception as e:
            print("Error:", e)
            response_code = 500

        self.send_response(response_code)
        self.send_header('Content-type', 'application/json')
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        if response:
            self.wfile.write(json.dumps(response).encode('utf-8'))


if __name__=='__main__':
    server = HTTPServer(('localhost',8000),RequestHandler)
    print('Server running on http://localhost:8000')
    server.serve_forever()
    