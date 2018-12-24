'''
Real-time midi processor, feeding relevant notes as time series into a single-layer GrFNN model. On
model changes, or user-triggered actions, this emits updates over sockets to localhost - a server
can pass these changes onto clients needing this information (i.e. js GUI/sonification tool).

@author T. Kaplan
'''

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mido
import numpy as np
import os
import queue
import re
import scipy.signal
import sys
import threading
import time

# This is messy, but for now simpler than packaging the project in full
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + '../web/')
from client import GrFNNClient, GrFNNAction
from gfnn import FrequencyType, FrequencyDist, ZParams, GrFNN

# Use the below sampling frequency to construct a time series from MIDI notes for GrFNN
FS = 160.0
DT = 1.0/FS

# As the MIDI notes might be infrequent impulses, scaling the velocity to ensure they make waves
MIDI_VEL_SCALE = 10.0

# GrFNN configuration - 250 oscillators in a gradient from 0.25Hz to 4.0Hz (linear spacing).
GRFNN_FROM_FREQ = 0.25
GRFNN_TO_FREQ = 4.00
GRFNN_DIM_IN = 250
F_DIST = FrequencyDist(GRFNN_FROM_FREQ, GRFNN_TO_FREQ, GRFNN_DIM_IN, FrequencyType.LINEAR)

# Constants for DTX502 MIDI channel, and corresponding notes
CHANNEL = 'DTX drums'
TOM_NOTES = {48, 47}
TOM_RIM_NOTE = 15
PEDAL_NOTE = 46
VALID_NOTES = TOM_NOTES | {TOM_RIM_NOTE, PEDAL_NOTE}

# Aggregate number of rim triggers in this window, the count will decide the action
RIM_TRIGGER_WINDOW = 0.75

class GrFNNMidiConsumer(threading.Thread):
    ''' This thread encapsulates a GrFNN, and client socket-based connection to whatever server is
    acting as a broker for model updates/user triggers for downstream processes '''

    def __init__(self):
        self._prev_time = None
        # NOTE: this lock isn't really necessary, but was used as a reminder, as originally the plan
        # was to have a separate thread persist GrFNN state in a useful form for ad-hoc analysis
        self._lock = threading.RLock()
        self._receive_queue = queue.Queue()
        self._stop_signal = threading.Event()
        self._captured_sequence = []
        # The GrFNN itself is constructed/initialised here, before main thread gets busy
        self._model = self.create_new_model()
        # The client socket-based connection, with convenient hooks for emitting updates/actions
        self._client = GrFNNClient()
        # This is handy, as sampling an empty initialized GrFNN will give complex vars
        self._zeros = np.zeros(F_DIST.dist.shape)
        # Track when a rim shot is triggered, and aggregate within the RIM_TRIGGER_WINDOW
        self._rimshots = 0
        self._first_rimshot_time = None
        super(GrFNNMidiConsumer, self).__init__()

    @staticmethod
    def create_new_model():
        ''' Convenient factory-esque method for refreshing this._model  '''
        return GrFNN(F_DIST, ZParams(), fs=FS)

    def run(self):
        # NOTE: uncomment if wanting to plot of input MIDI as time series
        #plt.plot(list(range(len(self._captured_sequence))), self._captured_sequence)

        # Prepare plot of GrFNN amplitudes
        px = F_DIST.dist
        plt.figure(figsize=(10,3))
        plt.plot(px, self._zeros)
        plt.plot(px, self._zeros)

        # Non-blocking retrieval from the queue of MIDI notes, which is added to by main thread
        self._prev_time = time.time()
        while not self._stop_signal.is_set():
            try:
                msg = self._receive_queue.get(block=False)
            except queue.Empty:
                # If we don't have a message, check the state of our rim shot action accumulator
                if self._rimshots and time.time() - self._first_rimshot_time > RIM_TRIGGER_WINDOW:
                    self._process_rim_action()
                    self._rimshots = 0
                    self._first_rimshot_time = None
            else:
                # Process message using GrFNN
                self._capture_message(msg)
        
        # Below is necessary to maintain our persistent plot, updated by event
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('z')
        plt.show()

    def stop(self):
        self._stop_signal.set()
        self.join()

    def receive(self, msg):
        self._receive_queue.put(msg)

    def _process_rim_action(self):
        action = GrFNNAction.rim_action(self._rimshots)

        if action != GrFNNAction.NONE:
            self._client.send_action(action)
        if action in (GrFNNAction.RESET, GrFNNAction.NEW_GRFNN):
            # In both cases, downstream will know how to handle the action - but here, we just want
            # to clear the model

            # Reset model and ensure that propagates downstream
            with self._lock:
                self._model = self.create_new_model()

            # Plot the reset amp curve:
            plt.gca().lines[0].set_ydata(self._zeros)
            plt.gca().lines[1].set_ydata(self._zeros)
            plt.gca().relim()
            plt.gca().autoscale_view()
            plt.pause(0.01)

    def _capture_message(self, msg):
        if msg.note == PEDAL_NOTE:
            self._client.send_action(GrFNNAction.NEW_TONES)

        elif msg.note == TOM_RIM_NOTE:
            # Aggregate rim shots, these are processed in _process_rim_action
            if not self._first_rimshot_time:
                self._first_rimshot_time = time.time()
            self._rimshots += 1

        elif msg.note in TOM_NOTES:
            # Construct a time series, by zero-padding since our last MIDI onset, informed by Fs/dt
            elapsed = msg.time - self._prev_time
            dts_since_last_note = int(np.floor(elapsed/DT))
            ts = np.zeros(dts_since_last_note+1)
            ts[-1] = msg.velocity*MIDI_VEL_SCALE 
            self._prev_time = msg.time

            # Inject time series into GrFNN 
            with self._lock:
                for val in ts:
                    self._model(val)

            z = abs(self._model.z)

            # Normalise to 0-1 range for downstream use, to identify most active oscillators
            z_norm = z/z.max()

            # Extract peaks, as downstream sonification won't want to sonify neighbour oscillators
            # with similar frequencies (e.g. wouldn't want 2.00025Hz AND 2.00050Hz)
            z_smooth = scipy.signal.savgol_filter(z, 51, 3)
            peaks = scipy.signal.argrelextrema(z_smooth, np.greater, order=50)[0]

            # Propagate updated model state downstream, list-ify for json compatibility
            self._client.send_heartbeat(z_norm.tolist(), peaks.tolist())

            # Plot the changing amp curve:
            plt.gca().lines[0].set_ydata(z)
            plt.gca().lines[1].set_ydata(z_smooth)
            plt.gca().relim()
            plt.gca().autoscale_view()
            plt.pause(0.01)

        # NOTE: add this back if tracking the complete time series since program start
        # If testing MIDI in as time series:
        #plt.gca().lines[0].set_ydata(self._captured_sequence)
        #plt.gca().lines[0].set_xdata(list(range(len(self._captured_sequence))))
        #
        #with self._lock:
        #    self._captured_sequence.extend(ts)


class MidiProcessor(object):
    ''' This provides a callback hook for rtmidi, filtering notes based on relevance, and
    timestamping them before passing them to a consumer which manages further processing '''

    def __init__(self, inport):
        self._consumer = GrFNNMidiConsumer()
        self._good_msg_type = re.compile('note_on')
        self._time = None
        inport.callback = self._handle_message

    def start(self):
        self._consumer.start()
        try:
            # Spin a busy loop, waiting for MIDI
            while True:
                time.sleep(.25)
        except KeyboardInterrupt:
            return
        except Exception as exc:
            # For now, just print exceptions, and let a user cancel out the process if necessary
            print(exc)

    def __del__(self):
        self._consumer.stop()

    def _handle_message(self, msg):
        curr_time = time.time()
        # Regex-match the message type (only care about 'note_on'), and check note number
        if self._good_msg_type.match(msg.type) and msg.note in VALID_NOTES and msg.velocity > 0:
            msg.time = curr_time
            self._consumer.receive(msg)

def process_midi():
    print('Available MIDI-in:', mido.get_input_names())
    mido.set_backend('mido.backends.rtmidi')

    inport = None
    try:
        inport = mido.open_input(CHANNEL)
    except Exception as exc:
        print(exc)
        sys.exit(1)

    hub = MidiProcessor(inport)
    try:
        hub.start()
    except Exception as exc:
        print(exc)
    finally:
        del hub
        inport.close()
        del inport

if __name__ == "__main__":
    process_midi()
