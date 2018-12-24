'''
@author T. Kaplan
'''
from flask import Flask, render_template, copy_current_request_context
from flask_socketio import SocketIO, emit
import os
import sys
import time

class GrFNNKey(object):
    ''' Encapsulates keys used for socket endpoints '''
    AMPS = "GrFNN_Amps"
    PEAKS = "GrFNN_Peaks"
    ACTION = "GrFNN_Action"
    DATA = "GrFNN_Data"

app = Flask(__name__)
app.config['SECRET_KEY'] = 'idmt4life'
socketio = SocketIO(app)

@socketio.on(GrFNNKey.DATA)
def handle_data(json):
    # Pass on data from GrFNN to clients
    socketio.emit(GrFNNKey.DATA, json)

@socketio.on(GrFNNKey.ACTION)
def handle_action(json):
    # Pass on action from user-input to clients
    print('Received action:', json)
    socketio.emit(GrFNNKey.ACTION, json)

@socketio.on('connected')
def handle_connection(json):
    print('Received connection confirmation: ' + str(json))

if __name__ == "__main__":
    socketio.run(app)

# NOTE: If client-side python (model) sockets fail, this code blocks Flask and uses its context
# and an emit hook to communicate with any other clients.
#
#sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../model/')
#import rt_midi_gfnn
## Create a function to pass into MIDI processer with communicates via socket
#def _emit_handle(data):
#    with app.test_request_context():
#        socketio.emit(GRFNN_SOCKET_KEY, data)
## Run MIDI polling process with current context to allow Flask-y stuff
#@copy_current_request_context
#def _midi_with_flask_ctx_faker():
#    rt_midi_gfnn.process_midi(_emit_handle)
#_midi_with_flask_ctx_faker()
#return "Listening for MIDI..."
