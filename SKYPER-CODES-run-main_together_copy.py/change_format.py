
# for set in ["train","valid","test"]:
#     inpath = "data/Wiki-whole/n-ary_"+set+".json"
#     outpath = "data/WikiPeople!=0/"+set+".txt"
#     with open(inpath,"r") as fin, open(outpath,"w") as fout:
#         fin = fin.read().strip().split("\n")
#
#         count = 0
#         for line in fin :
#
#             line2dict = eval(line)
#             relation = list(line2dict.keys())[0]
#             values = list(line2dict.values())
#             values.pop()
#             values = [ v for aList in values for v in aList ]
#             newList = [[v,"o"]for v in values]
#
#
#             newList = [ v for aList in newList for v in aList ]
#             newList.pop()
#             newList[1]=relation
#             out = " ".join(newList)+"\n"
#             print(out)


            # fout.write(out)

for set in ["train","valid","test"]:
    inpath = "data/Wiki-whole/n-ary_"+set+".json"
    outpath = "data/WikiPeople2!=0/"+set+".txt"
    with open(inpath,"r") as fin, open(outpath,"w") as fout:
        fin = fin.read().strip().split("\n")

        count = 0
        for line in fin :
            count += 1
            line2dict = eval(line)
            relations = list(line2dict.keys())
            relations.pop()
            values = list(line2dict.values())
            values.pop()

            realtion = relations[0]
            new_fact = []
            triple = [values[0][0], realtion, values[0][1],realtion+"_"+str(1)]

            # for i in range(len(relations)):
            #     value_relation_map = line2dict[relations[i]]
            #     for j in range(len(value_relation_map)):
            #         new_fact.append(value_relation_map[j])
            #         new_fact.append(relations[i]+"_"+str(i)+str(j))
            # print(new_fact)

            out = " ".join(triple)+"\n"
            fout.write(out)
# import torch
# import numpy
# pthfile = "E:\PycharmProjects\poincare-nary\data\JF17K-hinge-all\HINGE_128_600_100_0.0001"
# with open("vector.tsv","w") as fout:
#     net = torch.load(pthfile)
#     # print(net['Eh.weight'])
#     for vector in net['Eh.weight']:
#         # print(numpy.array(i.cpunumpy.array(i.cpu())()))
#         vector = numpy.array(vector.cpu())
#         print(vector)
#         for j in range(len(vector)):
#             if j!= len(vector)-1:
#                 fout.write(str(vector[j])+"\t")
#             else:
#                 fout.write(str(vector[j]) + "\n")

# with open("label.tsv","w") as fout:
#         for j in range(28645):
#                 fout.write(str(j) + "\n")

a = 27
print(a**-3)