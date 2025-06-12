import HopfieldNetwork as HN

startsJS = [HN.StartNeuron(1, 1, 1, 4), HN.StartNeuron(1, 2, 2, 1),
            HN.StartNeuron(1, 3, 3, 7), HN.StartNeuron(1, 4, 4, 7),
            HN.StartNeuron(2, 1, 2, 3), HN.StartNeuron(2, 2, 4, 7),
            HN.StartNeuron(2, 3, 1, 2), HN.StartNeuron(2, 4, 3, 8),
            HN.StartNeuron(3, 1, 4, 2), HN.StartNeuron(3, 2, 2, 2),
            HN.StartNeuron(3, 3, 1, 7), HN.StartNeuron(3, 4, 3, 2)]

ex1 = [HN.StartNeuron(1, 1, 1, 5), HN.StartNeuron(1, 2, 2, 8),
       HN.StartNeuron(1, 3, 3, 2), HN.StartNeuron(2, 1, 3, 7),
       HN.StartNeuron(2, 2, 1, 3), HN.StartNeuron(2, 3, 2, 9)]

ex2 = [HN.StartNeuron(1, 1, 1, 4), HN.StartNeuron(1, 2, 2, 3),
       HN.StartNeuron(1, 3, 3, 2), HN.StartNeuron(2, 1, 2, 1),
       HN.StartNeuron(2, 2, 1, 4), HN.StartNeuron(2, 3, 3, 4),
       HN.StartNeuron(3, 1, 3, 3), HN.StartNeuron(3, 2, 2, 2),
       HN.StartNeuron(3, 3, 1, 3), HN.StartNeuron(4, 1, 2, 3),
       HN.StartNeuron(4, 2, 3, 3), HN.StartNeuron(4, 3, 1, 1)]


myNet = HN.Network(ex2, 4, 3, 3)
for i in range(500):
    myNet.round()
outputList = [i.output()[0] for i in myNet.getStart()]
print(outputList)
print(myNet.time())
b = HN.BoltzMachine([i.output() for i in myNet.getStart()],
                    myNet.time())
temp = b.anneal()[1]
outputList = [i.output()[0] for i in temp.getStart()]
print(outputList)
print(temp.time())
b = HN.BoltzMachine([i.output() for i in myNet.getStart()],
                    myNet.time())
temp = b.anneal()[1]
outputList = [i.output()[0] for i in temp.getStart()]
print(outputList)
print(temp.time())
b = HN.BoltzMachine([i.output() for i in myNet.getStart()],
                    myNet.time())
temp = b.anneal()[1]
outputList = [i.output()[0] for i in temp.getStart()]
print(outputList)
print(temp.time())
