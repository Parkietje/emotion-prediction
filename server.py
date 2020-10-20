from http.server import BaseHTTPRequestHandler
import socketserver
import re
import predict
import screenshot

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
            # get file path
            file_path = str(self.path.split('/file/')[-1])
            print(file_path)
            valence, arousal = predict.predict(file_path)
            self.wfile.write(str("Valence: " + str(valence) + ";  Arousal: " + str(arousal)).encode())
            return
        
        # return prediction from screenshot
        if None != re.search('/screenshot', self.path):
            self._set_headers()
            valence, arousal = predict.predict(screenshot.screenshot())
            self.wfile.write(str("Valence: " + str(valence) + ";  Arousal: " + str(arousal)).encode())
            return

httpd = socketserver.ThreadingTCPServer(('', PORT),CustomHandler)

print("serving at port: " + str(PORT))
httpd.serve_forever()