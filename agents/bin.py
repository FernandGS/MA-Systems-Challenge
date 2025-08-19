import agentpy as ap

class Bin(ap.Agent):
    def setup(self):
        self.messages_exchanged = 0