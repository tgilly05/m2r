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


class HopfieldNeuron(Neuron):
    def __init__(self, job, op, machine, time, state=0, threshold=0):
        super().__init__(state)
        self.job = job
        self.op = op
        self.machine = machine
        self.time = time
        self.threshold = threshold

    def ac(self, network, weight=0.01):
        # the following sums are the shifts in start time based on
        # order contraints and machine use
        sum1 = 0  # the sum of interactions with neurons of the same job
        sum2 = 0  # the sum of interactions with neurons on the same machine
        for neu in network.getNeurons():
            # operation order constraint
            if neu.getJob() == self.job:
                if neu.getOp() == self.op+1:
                    sum1 -= self.schedule(self, neu)*weight*5
                elif neu.getOp() == self.op-1:
                    sum1 += self.schedule(neu, self)*weight*5
            # machine overlap constraint
            if neu.getMac() == self.machine:
                if neu.getJob() != self.job:
                    if self.state < neu.getState():
                        sum2 -= self.resource(self, neu)*weight
                    else:
                        sum2 += self.resource(neu, self)*weight
        N = sum1 + sum2 + self.state
        # we hold this N for synchronus updates
        if N < self.threshold:
            self.hold = self.threshold
        else:
            self.hold = N

    def schedule(self, n1, n2):
        N = n1.getState() + n1.getTime() - n2.getState()
        if N <= 0:
            return 0
        return N

    def resource(self, n1, n2):
        N = n1.getState() + n1.getTime() - n2.getState()
        if N <= 0:
            return 0
        return N

    def getTime(self):
        return self.time

    def output(self):
        return (round(self.state),
                (self.job, self.op, self.machine),
                self.time)

    def getJob(self):
        return self.job

    def getOp(self):
        return self.op

    def getMac(self):
        return self.machine

    def getEnd(self):
        return self.state + self.time


class StartNeuron(Neuron):
    def __init__(self, job, op, machine, time, state=0, threshold=0):
        super().__init__(state)
        self.job = job
        self.op = op
        self.machine = machine
        self.time = time
        self.threshold = threshold

    def ac(self, network, weight=0.01):
        if self.op != 1:
            self.threshold = network.getStart()[
                network.char()*(self.job-1)+self.op-2].getEnd()
        sum1 = 0
        for pair in network.getSch():
            c1 = (pair.n1.getLocation() == self.getLocation())
            c2 = (pair.n2.getLocation() == self.getLocation())
            if c1:
                sum1 += pair.getState()[1]*weight*5
            elif c2:
                sum1 += pair.getState()[0]*weight*5
        sum2 = 0
        for pair in network.getRes():
            c1 = (pair.n1.getLocation() == self.getLocation())
            c2 = (pair.n2.getLocation() == self.getLocation())
            if c1:
                if pair.getState()[2]:
                    sum2 += pair.getState()[1]*weight
                else:
                    sum2 += pair.getState()[0]*weight
            elif c2:
                if pair.getState()[2]:
                    sum2 += pair.getState()[0]*weight
                else:
                    sum2 += pair.getState()[1]*weight
        N = sum1 + sum2 + self.state
        if N <= self.threshold:
            self.hold = self.threshold
            return self.threshold
        self.hold = N
        return N

    def output(self):
        return (round(self.state),
                (self.job, self.op, self.machine),
                self.time)

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
        self.state = (state, state)
        self.n1 = n1
        self.n2 = n2
        self.location = (n1.getLocation(), n2.getLocation())

    def ac(self, network):
        # update incoming neuron states
        ind1 = network.char()*(self.location[0][0]-1) + self.location[0][1] - 1
        ind2 = network.char()*(self.location[1][0]-1) + self.location[1][1] - 1
        self.n1, self.n2 = network.getStart()[ind1], network.getStart()[ind2]
        # applies the activation formula
        N = self.n1.getState() + self.n1.getTime() - self.n2.getState()
        if N <= 0:
            self.hold = (0, 0)
            return (0, 0)
        self.hold = (N, -1*N)
        return (N, -1*N)


class ResourceNeuron(Neuron):
    def __init__(self, n1, n2, state=0):
        self.state = (state, state, True)
        self.n1 = n1
        self.n2 = n2
        self.location = (n1.getLocation(), n2.getLocation())

    def ac(self, network):
        # update input neurons
        ind1 = network.char()*(self.location[0][0]-1) + self.location[0][1] - 1
        ind2 = network.char()*(self.location[1][0]-1) + self.location[1][1] - 1
        self.n1, self.n2 = network.getStart()[ind1], network.getStart()[ind2]
        # order neurons by start time
        if self.n1.getState() < self.n2.getState():
            first, last = self.n1, self.n2
        else:
            first, last = self.n2, self.n1
        # applies the activation formula
        N = last.getEnd() - first.getState()
        if N <= first.getTime() + last.getTime():
            self.hold = (N, -1*N, first == self.n1)
            return self.hold
        self.hold = (0, 0, first == self.n1)
        return self.hold


class Network:
    def __init__(self, Starts, job, op, mac):
        self.timespan = 0
        self.character = mac
        self.startNeurons = Starts
        self.scheduleNeurons = []
        for i in range(job):
            for j in range(1, op):
                self.scheduleNeurons.append(
                    ScheduleNeuron(self.startNeurons[i*(op)+j-1],
                                   self.startNeurons[i*(op)+j])
                )
        self.resourceNeurons = []
        for k in range(1, mac+1):
            for i in range(job*op):
                if self.startNeurons[i].getLocation()[2] == k:
                    for j in range(i+1, job*op):
                        if self.startNeurons[j].getLocation()[2] == k:
                            self.resourceNeurons.append(
                                ResourceNeuron(self.startNeurons[i],
                                               self.startNeurons[j])
                            )

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

    def char(self):
        return self.character

    def time(self):
        for i in self.startNeurons:
            if i.getEnd() > self.timespan:
                self.timespan = i.getEnd()
        return round(self.timespan)


class Hopfield():
    def __init__(self, Starts, job, op, mac):
        self.timespan = 0
        self.character = op
        self.neurons = Starts

    def optimise(self):
        for i in range(5000):
            self.round()

    def round(self):
        for n in self.neurons:
            n.ac(self)
        for n in self.neurons:
            n.update()
        # normalise
        first = 1000
        for i in self.neurons:
            if i.getState() < first:
                first = i.getState()
        for i in self.neurons:
            i.setState(i.getState()-first)

    def getNeurons(self):
        return self.neurons

    def char(self):
        return self.character

    def time(self):
        for i in self.neurons:
            if i.getEnd() > self.timespan:
                self.timespan = i.getEnd()
        return round(self.timespan)
