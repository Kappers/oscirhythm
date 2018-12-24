'''
@author T. Kaplan
'''

from socketIO_client import SocketIO, LoggingNamespace

from server import GrFNNKey

class GrFNNAction(object):
    ''' Possible user-triggered actions when using interactive GrFNN (enum-esque) '''

    NONE = ""
    # Reset any downstream state
    RESET = "RESET"
    # Create a new GrFNN downstream, in parallel to existing
    NEW_GRFNN = "NEW_GRFNN"
    # Lock the rhythm on the active GrFNN downstream
    LOCK = "LOCK"
    # Pedal trigger, change tones downstream for active GrFNN
    NEW_TONES = "NEW_TONES"

    @classmethod
    def rim_action(cls, rimshots):
        ''' Return action corresponding to given number of rimshot hits in a short window '''
        if rimshots is 1:
            return cls.LOCK
        elif rimshots is 2:
            return cls.NEW_GRFNN
        elif rimshots is 3:
            return cls.RESET
        else:
            return cls.NONE

class GrFNNClient(object):
    ''' Encapsulates a client socket connection, and a hook for emitting actions/model updates '''

    def __init__(self, host='127.0.0.1', port=5000):
        self._socket = SocketIO(host, port, LoggingNamespace)

    def send_action(self, action_flag):
        # Use this to send only an action, as sometimes it isn't related to a model change
        self._socket.emit(GrFNNKey.ACTION, action_flag);

    def send_heartbeat(self, normalised_amps, peaks):
        # Use this to send normalised amplitude array and peaks (active oscillators)
        self._socket.emit(GrFNNKey.DATA, {
            GrFNNKey.AMPS: normalised_amps,
            GrFNNKey.PEAKS: peaks
        })
