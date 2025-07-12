from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from neural_network_design import OCRNeuralNetwork
from ocr import nn



class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        response_code = 200
        response = ''

        try:
            content_length = int(self.headers.get('Content-Length'))
            content = self.rfile.read(content_length)
            payload = json.loads(content)

            if payload.get('train'):
                for item in payload['trainArray']:
                    x=item['y0']
                    label=item['label']
                    nn.train_single_sample(x,label)
                nn.save()
                   

            elif payload.get('predict'):
                try:
                    print('Predict request received!')
                    print('Payload length:', len(payload['image']))
                    print('Payload length:', len(payload['image']))
                    print("Sample values:", payload['image'][:10])
                    response = {
                        'type': 'test',
                        'result': nn.predict((payload['image']))
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
    