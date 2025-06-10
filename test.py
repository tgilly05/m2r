import HopfieldNetwork as HN

sn_test = HN.StartNeuron(1, 1, 1, 5, state=4)
sn2 = HN.StartNeuron(1, 2, 1, 3, state=0)

sc = HN.ScheduleNeuron(sn_test, sn2)
rn = HN.ResourceNeuron(sn_test, sn2)

timelist = [1, 2,
            2, 3,
            5, 2,
            4, 1,
            4, 2,
            3, 1]
myNet = HN.Network(timelist, 3, 2, 2)

rounds = []
rounds.append([i.getState() for i in myNet.getStart()])
myNet.round()
rounds.append([i.getState() for i in myNet.getStart()])
myNet.round()
rounds.append([i.getState() for i in myNet.getStart()])
myNet.round()
myNet.round()
myNet.round()
myNet.round()
myNet.round()
rounds.append([i.getState() for i in myNet.getStart()])
print(rounds[0])
print(rounds[1])
print(rounds[2])
print(rounds[3])
