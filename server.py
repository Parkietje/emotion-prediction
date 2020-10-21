from http.server import BaseHTTPRequestHandler
import socketserver
import re
import predict
import portrait
import json
import os
from KNN import knn

PORT = 9000

class CustomHandler(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        
        #
        # return prediction for an existing input file
        #
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

        #
        # return prediction from portrait
        #
        if None != re.search('/screenshot', self.path):
            self._set_headers()            
            # make portrait
            try:
                img = portrait.capture()
            except Exception as e:
                response = {'error': str(e)}
                self.wfile.write(json.dumps(response, indent=2).encode('utf-8'))
                return
            valence, arousal = predict.predict(img)
            emotion = knn.predict_emotion(valence, arousal)
            # make response object
            response = {'valence': valence, 'arousal':arousal, 'emotion':emotion}
            self.wfile.write(json.dumps(response, indent=2).encode('utf-8'))
            return
        
        #
        #  train KNN algorithm
        #
        if None != re.search('/train', self.path):
            self._set_headers()
            # parse train label
            label = str(self.path.split('/train/')[-1])
            EMOTIONS = {-2: 'angry', -1: 'annoyed', 0: 'neutral', 1: 'content', 2: 'joyful'}
            if not label in EMOTIONS.values():
                response = {'error': 'label not recognized'+label}
                self.wfile.write(json.dumps(response, indent=2).encode('utf-8'))
                return
            # get prediction from portrait
            try:
                img = portrait.capture()
            except Exception as e:
                response = {'error': str(e)}
                self.wfile.write(json.dumps(response, indent=2).encode('utf-8'))
                return
            valence, arousal = predict.predict(img)
            # append new feature to train data
            with open(os.path.join(os.getcwd(),'KNN/train_data.csv'), 'a') as f:
                f.write('\n'+ str(valence) + ',' + str(arousal) + ',' + label)
            # make response object
            response = {'valence': valence, 'arousal':arousal, 'label':label}
            self.wfile.write(json.dumps(response, indent=2).encode('utf-8'))
            return

httpd = socketserver.ThreadingTCPServer(('', PORT),CustomHandler)

print("serving at port: " + str(PORT))
httpd.serve_forever()