import math as mt
import numpy as np


class Neuron:
    def __init__(self, state=0):
        self.state = state
        self.hold = state

    def ac(self, *args, **kwds):
        pass

    def getState(self):
        return self.state

    def setState(self, newState):
        self.state = newState

    def update(self):
        self.state = self.hold


class StartNeuron(Neuron):
    def __init__(self, job, op, machine, time, state=0, threshold=0):
        super().__init__(state)
        self.job = job
        self.op = op
        self.machine = machine
        self.time = time
        self.threshold = threshold

    def ac(self, network, weight=0.1):
        if self.op != 1:
            mul = network.getChar()
            ind = (mul[0]*(self.job-1) + mul[2]*(self.machine-1) - 1)
            self.threshold = network.getStart()[ind].getEnd()
        sum1 = 0
        for pair in network.getSch():
            sum1 += pair.getState()*weight
        sum2 = 0
        for pair in network.getRes():
            sum2 += pair.getState()*weight
        N = sum1 + sum2 + self.state
        if N <= self.threshold:
            self.hold = self.threshold
            return self.threshold
        self.hold = N
        return N

    def getEnd(self):
        return self.state + self.time

    def getTime(self):
        return self.time

    def getLocation(self):
        return (self.job, self.op, self.machine)

    def setThresh(self, thresh):
        self.threshold = thresh


class ScheduleNeuron(Neuron):
    def __init__(self, n1, n2, state=0):
        super().__init__(state)
        self.n1 = n1
        self.n2 = n2
        self.location = (n1.getLocation(), n2.getLocation())

    def ac(self, network):
        # update incoming neuron states
        mul = network.getChar()
        ind1 = 0
        ind2 = 0
        for i in range(3):
            ind1 += mul[i]*(self.location[0][i] - 1)
            ind2 += mul[i]*(self.location[1][i] - 1)
        self.n1, self.n2 = network.getStart()[ind1], network.getStart()[ind2]
        # orders neurons by operation order
        if self.location[0][1] < self.location[1][1]:
            first, last = self.n1, self.n2
        else:
            first, last = self.n2, self.n1
        # applies the activation formula
        N = first.getState() + first.getTime() - last.getState()
        if N < 0:
            self.hold = 0
            return 0
        self.hold = N
        return N


class ResourceNeuron(Neuron):
    def __init__(self, n1, n2, state=0):
        super().__init__(state)
        self.n1 = n1
        self.n2 = n2
        self.location = (n1.getLocation(), n2.getLocation())

    def ac(self, network):
        # update input neurons
        mul = network.getChar()
        ind1 = 0
        ind2 = 0
        for i in range(3):
            ind1 += mul[i]*(self.location[0][i] - 1)
            ind2 += mul[i]*(self.location[1][i] - 1)
        self.n1, self.n2 = network.getStart()[ind1], network.getStart()[ind2]
        # order neurons by start time
        if self.n1.getState() < self.n2.getState():
            first, last = self.n1, self.n2
        else:
            first, last = self.n2, self.n1
        # applies the activation formula
        N = last.getEnd() - first.getState()
        if N < first.getTime() + last.getTime():
            self.hold = N
            return N
        self.hold = 0
        return 0


class Network:
    def __init__(self, timeList, job, op, machine):
        self.character = (machine*op, machine * (op - 1), machine - 1)
        self.startNeurons = [StartNeuron(mt.floor(i/(op*machine)) + 1,
                                         (mt.floor(i/machine) % op) + 1,
                                         (i % machine) + 1,
                                         timeList[i])
                             for i in range(len(timeList))]
        for i in range(len(timeList)):
            if (mt.floor(i/machine) % op) != 0:
                self.startNeurons[i].setThresh(self.startNeurons[
                    i-machine].getEnd())
        self.resourceNeurons = []
        for k in range(machine):
            for i in range(job * op):
                for p in range(i + 1, job * op):
                    self.resourceNeurons.append(ResourceNeuron(
                        self.startNeurons[(i*machine+k)],
                        self.startNeurons[(p*machine+k)]))
        self.scheduleNeurons = []
        for i in range(job):
            for j in range(op * machine):
                for a in range(j + 1, op * machine):
                    self.scheduleNeurons.append(ScheduleNeuron(
                        self.startNeurons[(i*(op * machine)+j)],
                        self.startNeurons[(i*(op * machine)+a)]))

    def round(self):
        for n in self.startNeurons:
            n.ac(self)
        for n in self.scheduleNeurons:
            n.ac(self)
        for n in self.resourceNeurons:
            n.ac(self)
        for n in self.startNeurons:
            n.update()
        for n in self.scheduleNeurons:
            n.update()
        for n in self.resourceNeurons:
            n.update()
        # normalise back to first op at 0
        first = 100
        for i in self.startNeurons:
            if i.getState() < first:
                first = i.getState()
        for i in self.startNeurons:
            i.setState(i.getState()-first)

    def getStart(self):
        return self.startNeurons

    def getRes(self):
        return self.resourceNeurons

    def getSch(self):
        return self.scheduleNeurons

    def getChar(self):
        return self.character
