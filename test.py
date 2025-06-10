import HopfieldNetwork as HN

sn_test = HN.StartNeuron(1, 1, 1, 5, state=4)
sn2 = HN.StartNeuron(1, 2, 1, 3, state=0)

sc = HN.ScheduleNeuron(sn_test, sn2)
rn = HN.ResourceNeuron(sn_test, sn2)

timelist = [3, 2,
            4, 2,
            1, 6,
            2, 5,
            7, 2,
            1, 3]
myNet = HN.Network(timelist, 3, 2, 2)

print(sn_test.getLocation())
print(sn_test.getEnd())
print(sc.ac())
print(rn.ac())
print(len(myNet.getRes()))
