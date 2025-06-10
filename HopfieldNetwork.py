import math as mt
import numpy as np


class Neuron:
    def __init__(self, state=0):
        self.state = state

    def ac(self, *args, **kwds):
        pass

    def getState(self):
        return self.state


class StartNeuron(Neuron):
    def __init__(self, job, op, machine, time, state=0, threshold=0):
        super().__init__(state)
        self.job = job
        self.op = op
        self.machine = machine
        self.time = time
        self.threshold = threshold

    def ac(self, network, weight=0.1):
        N = np.dot(network.getState) + self.state
        return N

    def getEnd(self):
        return self.state + self.time

    def getTime(self):
        return self.time

    def getLocation(self):
        return (self.job, self.op, self.machine)


class ScheduleNeuron(Neuron):
    def __init__(self, n1, n2, state=0):
        super().__init__(state)
        self.location = (n1.getLocation(), n2.getLocation())

    def ac(self, n1, n2):
        # orders neurons
        if n1.getState() < n2.getState():
            first, last = n1, n2
        else:
            first, last = n2, n1
        # applies the activation formula
        N = first.getState() + first.getTime() - last.getState()
        if N < 0:
            self.state = 0
            return 0
        self.state = N
        return N


class ResourceNeuron(Neuron):
    def __init__(self, n1, n2, state=0):
        super().__init__(state)
        self.location = (n1.getLocation(), n2.getLocation())

    def ac(self, n1, n2):
        # first order neurons by start time
        if n1.getState() < n2.getState():
            first, last = n1, n2
        else:
            first, last = n2, n1
        if first.getEnd() <= last.getEnd():
            return
        # applies the activation formula
        N = last.getEnd() - first.getState()
        if N < first.getEnd() + last.getTime():
            self.state = N
            return N
        self.state = 0
        return 0


class Network:
    def __init__(self, timeList):
        self.startNeurons = [StartNeuron(0,
                                         mt.floor(i/9) + 1,
                                         mt.floor(i/3) + 1,
                                         (i % 3) + 1,
                                         timeList[i])
                             for i in range(len(timeList))]
        self.resourceNeurons = []
        self.scheduleNeurons = []

    def getStart(self):
        return self.startNeurons

    def getRes(self):
        return self.resourceNeurons

    def getSch(self):
        return self.scheduleNeurons
