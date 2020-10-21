from http.server import BaseHTTPRequestHandler
import socketserver
import re
import predict
import screenshot
import json

PORT = 9090

class CustomHandler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
    
        # return prediction for an existing input file
        if None != re.search('/file/*', self.path):
            self._set_headers()
            
            # parse file path
            file_path = str(self.path.split('/file/')[-1])

            # get prediction for file
            valence, arousal = predict.predict(file_path)

            # make response object
            response = {'valence': valence, 'arousal':arousal}
            self.wfile.write(json.dumps(response, indent=2).encode('utf-8'))
            return
        
        # return prediction from screenshot
        if None != re.search('/screenshot', self.path):
            self._set_headers()
            
            # get prediction from screenshot
            valence, arousal = predict.predict(screenshot.screenshot())
            
            # make response object
            response = {'valence': valence, 'arousal':arousal}
            self.wfile.write(json.dumps(response, indent=2).encode('utf-8'))
            return

httpd = socketserver.ThreadingTCPServer(('', PORT),CustomHandler)

print("serving at port: " + str(PORT))
httpd.serve_forever()