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

ex3 = [HN.HopfieldNeuron(1, 1, 1, 5), HN.HopfieldNeuron(1, 2, 2, 8),
       HN.HopfieldNeuron(1, 3, 3, 2), HN.HopfieldNeuron(2, 1, 3, 7),
       HN.HopfieldNeuron(2, 2, 1, 3), HN.HopfieldNeuron(2, 3, 2, 9)]

ex4 = [HN.HopfieldNeuron(1, 1, 1, 4), HN.HopfieldNeuron(1, 2, 2, 3),
       HN.HopfieldNeuron(1, 3, 3, 2), HN.HopfieldNeuron(2, 1, 2, 1),
       HN.HopfieldNeuron(2, 2, 1, 4), HN.HopfieldNeuron(2, 3, 3, 4),
       HN.HopfieldNeuron(3, 1, 3, 3), HN.HopfieldNeuron(3, 2, 2, 2),
       HN.HopfieldNeuron(3, 3, 1, 3), HN.HopfieldNeuron(4, 1, 2, 3),
       HN.HopfieldNeuron(4, 2, 3, 3), HN.HopfieldNeuron(4, 3, 1, 1)]

task1 = [HN.HopfieldNeuron(1, 1, 1, 4), HN.HopfieldNeuron(1, 2, 2, 1),
         HN.HopfieldNeuron(1, 3, 3, 7), HN.HopfieldNeuron(1, 4, 4, 7),
         HN.HopfieldNeuron(2, 1, 2, 3), HN.HopfieldNeuron(2, 2, 4, 7),
         HN.HopfieldNeuron(2, 3, 1, 2), HN.HopfieldNeuron(2, 4, 3, 8),
         HN.HopfieldNeuron(3, 1, 4, 2), HN.HopfieldNeuron(3, 2, 2, 2),
         HN.HopfieldNeuron(3, 3, 1, 7), HN.HopfieldNeuron(3, 4, 3, 2)]

myNet = HN.Hopfield(task1, 3, 4, 4)
for i in range(5000):
    myNet.round()
outputList = [i.output()[0] for i in myNet.getNeurons()]
print(outputList)
print(myNet.time())
