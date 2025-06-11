import HopfieldNetwork as HN

sn_test = HN.StartNeuron(1, 1, 1, 5, state=4)
sn2 = HN.StartNeuron(1, 2, 1, 3, state=0)

sc = HN.ScheduleNeuron(sn_test, sn2)
rn = HN.ResourceNeuron(sn_test, sn2)

timelist = [2, 0,
            0, 2,
            0, 1,  # job 1
            0, 2,
            0, 1,
            2, 0,  # job 2
            1, 0,
            4, 0,
            1, 0]  # job 3


starts = [HN.StartNeuron(1, 1, 1, 4),
          HN.StartNeuron(1, 2, 2, 1),
          HN.StartNeuron(1, 3, 3, 7),
          HN.StartNeuron(1, 4, 4, 7),
          HN.StartNeuron(2, 1, 2, 3),
          HN.StartNeuron(2, 2, 4, 7),
          HN.StartNeuron(2, 3, 1, 2),
          HN.StartNeuron(2, 4, 3, 8),
          HN.StartNeuron(3, 1, 4, 2),
          HN.StartNeuron(3, 2, 2, 2),
          HN.StartNeuron(3, 3, 1, 7),
          HN.StartNeuron(3, 4, 3, 2)]

schedules = [HN.ScheduleNeuron(starts[0], starts[1]),
             HN.ScheduleNeuron(starts[1], starts[2]),
             HN.ScheduleNeuron(starts[2], starts[3]),
             HN.ScheduleNeuron(starts[4], starts[5]),
             HN.ScheduleNeuron(starts[5], starts[6]),
             HN.ScheduleNeuron(starts[6], starts[7]),
             HN.ScheduleNeuron(starts[8], starts[9]),
             HN.ScheduleNeuron(starts[9], starts[10]),
             HN.ScheduleNeuron(starts[10], starts[11])]

rescourse = [HN.ResourceNeuron(starts[0], starts[6]),
             HN.ResourceNeuron(starts[0], starts[10]),
             HN.ResourceNeuron(starts[6], starts[10]),
             HN.ResourceNeuron(starts[1], starts[4]),
             HN.ResourceNeuron(starts[1], starts[9]),
             HN.ResourceNeuron(starts[4], starts[9]),
             HN.ResourceNeuron(starts[2], starts[7]),
             HN.ResourceNeuron(starts[2], starts[11]),
             HN.ResourceNeuron(starts[7], starts[11]),
             HN.ResourceNeuron(starts[3], starts[5]),
             HN.ResourceNeuron(starts[3], starts[8]),
             HN.ResourceNeuron(starts[5], starts[8])]

myNet = HN.Network(starts, schedules, rescourse)

rounds = []
rounds.append([i.output() for i in myNet.getStart()])
myNet.round()
rounds.append([i.output() for i in myNet.getStart()])
for i in range(5000):
    myNet.round()
rounds.append([i.output() for i in myNet.getStart()])
outputList = [i.output()[0] for i in myNet.getStart()]
print(rounds[0])
print(rounds[1])
print(rounds[-1])
print(outputList)
