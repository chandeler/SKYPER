import json
import os
import numpy as np

from load_data import Data_whole_and_split

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch as torch
# path = "data/WikiPeople!=0/n-ary_test.json"
# with open(path,"r") as fin:
#     lines = fin.readlines()
#
# with open("data/WikiPeople3/test-1.txt",'w') as fout:
#     for line in lines:
#         line = eval(line)
#         if line['N'] == 3:
#             fout.write(line["H"]+" "+line["R"]+" "+line["T"]+"\n")

# import torch
# data = [[10167, 84, 5233, 85, 267, 86, 5241],
# [3159, 30, 572, 31, 573, 32, 574],
# [4047, 30, 572, 31, 573, 32, 574]]
#
# data1= [[10167, 84, 5233, 85, 267, 86, 5241],
# [3159, 30, 572, 31, 573, 32, 574],
# ]
# data2 =[1,2,3]
# data3 = [3,5]
# print(data2+data3)
# d = data2*5
# print("d",np.array(data2).repeat(2))
#
# data += [ [d[2],str(d[1])+"_reverse",d[0]]+[item for item in d[3:]] for d in data]
# print(data)
#
# M = torch.Tensor([[1,2,3],[2,3,4]])
# print(M.size())
# x = torch.Tensor([
#     [[1,2],[1,3],[1,119]],
#     [[1,7],[1,2],[3,2]]])
# x1 = torch.Tensor([
#     [[1,2,10],[1,3,5],[1,119,8]],
#     [[1,7,10],[1,2,10],[3,2,8]]])
# print(x1.T)
# print(x.size())
# b = torch.matmul(x,M)
# print(b)
# b_norm = torch.norm(b,dim= -1,keepdim=True)
# print(b_norm)
# print(b/b_norm)
# print(40/2)
# a = [i for i in range(10)]
# print(a)

# relations = sorted(list(set([d[i] for i in range(1,len(data[0]),2) for d in data])))

# entitiess = sorted(list(set([d[i] for i in range(0,len(data[0]),2) for d in data])))
# print(entitiess)
# entitiess_ids = {entitiess[i]:i for i in range(len(entitiess))}
# # datas_ids = [d[i] for i in range(len(data[0])) for d in data]
# data_idxs =[]
# for d in data:
#     dataid_2_relationssid = []
#     for i in range(len(d)):
#         if i % 2 == 1:
#             dataid_2_relationssid.append(relations_ids[d[i]])
#         else:
#             dataid_2_relationssid.append(entitiess_ids[d[i]])
#     data_idxs.append(tuple(dataid_2_relationssid))
# print(data_idxs)
# import numpy as np
# a = [1,2,3,4,5]
# a = np.array(range(2,len(a),2))
#
#
# b = np.random.choice(a)
# print(b)

# import pickle
# with open ("data/FilteredJF17K/dictionaries_and_facts.bin","rb") as fin:
#     dictionaries_and_facts = pickle.load(fin)
#     train = fin["train_factss"]
#     print(train)


# with open("data/JF17K4-811-4=0/train.txt", "w") as fout:
#
#     with open("data/JF17K4-811-4=0v1/train.txt", "r") as fin:
#         for line in fin:
#             empty = []
#             quintuple = line.split()
#             # temp = quintuple[0]
#             # quintuple[0] = quintuple[1]
#             # quintuple[1] = temp
#             # temp = quintuple[3]
#             # quintuple[3] = "O"
#             # quintuple.append(temp)
#             # print(quintuple)
#             # fout.write(" ".join(quintuple)+"\n")
#             r = quintuple[0]
#             e1 = quintuple[1]
#             e2 = quintuple[2]
#             e3 = quintuple[3]
#             e4 = quintuple[4]
#             empty.append(e1)
#             empty.append(r)
#             empty.append(e2)
#             empty.append("O")
#             empty.append(e3)
#             empty.append("O")
#             empty.append(e4)
#             fout.write(" ".join(empty) + "\n")



# a = torch.Tensor(range(12)).resize(3,4)
# print(a)
#
# b = torch.randint(20,[12]).resize(3,4)
# print(b)
# print(torch.min(a,b))
# a = torch.Tensor([[1,8,13]])
# b = torch.Tensor([2])
# c = torch.Tensor([3])
# d = torch.cat((a,b,c))
# print(d)
# print(d[-1])
# d[1:]  = d[1:] *5
# print(d[1: ])
# k = torch.randn(5,3,4)
# print(k)
# k[:,1:,:]=k[:,1:,:]*50.0
# print(k)
#
# m = range(0,10,2)
# for i in m:
#     print(i)
# x = torch.randn(128, 50, 20) # 输入的维度是（128，20）
# m = torch.nn.Linear(20, 30) # 20,30是指维度
# output = m(x)
# print('m.weight.shape:\n ', m.weight.shape)
# print('m.bias.shape:\n', m.bias.shape)
# print('output.shape:\n', output.shape)
# W = torch.randn(51,100,50)
#
#
# a = torch.randn(128,51,100)
#
#
# print((torch.mm(a,W)))

# weight_for_head = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (1000, 50,100)),
#                                                             dtype=torch.float32, requires_grad=True, device="cuda"))
# print(weight_for_head[torch.tensor([1,3,5])].shape)
#
# weight = weight_for_head[torch.tensor([1,3,5])]
#
# embedding = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (10000, 100)),
#                                                             dtype=torch.float32, requires_grad=True, device="cuda"))
#
# entities = [embedding[torch.tensor([3,7])],embedding[torch.tensor([7,10])],embedding[torch.tensor([3,10])]]
# print(entities)


# a = 0.1*torch.randint(0,5,(128,51,50,100)).reshape(-1,50,100)
# print(a)
# b = 0.1*torch.randint(0,5,(128,51,100)).reshape(-1,100).unsqueeze(-1)
# print(b)
# # print(b.unsqueeze(-1))
# print(torch.matmul(a,b))
# print(torch.matmul(a,b).squeeze())
# print(torch.matmul(a,b).squeeze().reshape(128,51,-1).shape)

# import torch
# print(torch.__version__)  #注意是双下划线
#
# print(torch.version.cuda)
# b = [1,3,18,0,15,0,16]
# len  = len(b)
# a = [2]+list(range(2,len,2))
# print(len,a)
# # with open("data/JF17K-hinge-all/test_new.txt","w") as fout:
# relations = []
# relations_2 = []
# with open("data/JF17K-hinge-all/train.txt", "r") as fin,open("data/JF17K-hinge-all/test.txt", "r") as fin2:
#     fin = fin.read().strip().split("\n")
#     for i,fact in enumerate(fin):
#         fact_new_format = []
#
#         facts = fact.split()
#
#         relation = facts[1]
#         if relation not in relations:
#
#             print(relation)
#             relations.append(relation)
#     print(len(relations))
#
#     fin = fin2.read().strip().split("\n")
#     for i, fact in enumerate(fin):
#         fact_new_format = []
#
#         facts = fact.split()
#         relation = facts[1]
#         if relation not in relations:
#             # print(relation)
#             relations.append(relation)
#
#         # fact_new_format.append(facts[1])
#         # fact_new_format.append(facts[0]+"1")
#         # for item in facts[2:]:
#         #     fact_new_format.append(item)
#         #     fact_new_format.append("O")
#         # fact_new_format.pop()
#         # fout.write("\t".join(fact_new_format)+"\n")
#         # print("new",i,fact_new_format)
#
# print(len(relations))

# d = Data_whole_and_split("data/JF17K-hinge-all/")
#
# train_relations = d.train_relations
# relations = d.relations
#
# for relation in relations:
#     if relation not in train_relations:
#         print(relation)
#
# print("hello")

# with open("data/Wiki-hinge/train.txt") as fin,open("data/Wiki-hinge/valid.txt") as fin1:
#     fin = fin.read().strip().split("\n")
#     relations = []
#     entities = []
#     for line in fin:
#         line = line.split(" ")
#         relation  = line[1]
#         if relation not in relations:
#             relations.append(relation)
#
#         for i  in range(0,len(line),2):
#             entities.append(line[i])
#
#         # print( line)
#     entities = list(set(entities))
#         # entities = entities+[ent for ent in line if ent not in entities]
#     print(len(entities))
#
#     fin1 = fin1.read().strip().split("\n")
#
#     for line in fin1:
#         # relations = []
#         line = line.split(" ")
#         relation = line[1]
#         if relation not in relations:
#             relations.append(relation)
#             print(relation)
#         for i  in range(0,len(line),2):
#             if line[i] not in entities:
#                 print(line[i])

        # print( line)

    #
# with open("data/Wiki-with-repeat/valid.txt","r") as fin,open("data/Wiki-without-repeat/train_triple.txt","w") as fout:
#     fin = fin.read().strip().split("\n")
#     print(len(fin))
#     fin = list(set(fin))
#     print(len(fin))
#     count = 0
#     for line in fin :
#         line2list = line.split(" ")
#
#         if len(line2list)>3:
#             count += 1
#             print(count,line2list[:3])
#             fout.write(" ".join(line2list[:3])+"\n")

# with open("data/Wiki-without-repeat/test.txt","r") as fin:
#     fin = fin.read().strip().split("\n")
#     print(len(fin))
#     fin = list(set(fin))
#     print(len(fin))
#     count = 0
#     for line in fin :
#         line2list = line.split(" ")
# #
#         if len(line2list)>3:
#             count += 1
#             print(count)
# #             fout.write(" ".join(line2list[:3])+"\n")

# a = torch.randn((10,50))
# print(a[1,3],a[5,4])
# print(a[1,3]+a[5,4])

# for i in ([1]+list(range(0,6,2))):
#     print(i)
# a = np.random.randint(10000)
# print(a)
#
# import torch
# torch.cuda.empty_cache()
# # 计算一下总内存有多少。
# print(torch.cuda.is_available())
# total_memory = torch.cuda.get_device_properties(0).total_memory
# # 占用全部显存:
# tmp_tensor = torch.empty(int(total_memory), dtype=torch.int8, device='cuda')




# for set in ["train","valid","test"]:
#     in_dataset_path = "data/JF17K/"
#     out_dataset_path = "data/JF17K4!=0/"
#     in_path = in_dataset_path+set+".txt"
#     out_path = out_dataset_path+set+".txt"
#     with open(in_path) as f, open(out_path, "w") as fout:
#         lines = f.readlines()
#         for line in lines:
#             fact = line.split()
#             # print(fact)
#             length = len(fact)
#             if length==5:
#                 relation = fact[0]
#                 entities = fact[1:]
#                 new_format_fact = []
#                 for i in range(len(entities)):
#                     new_format_fact.append(entities[i])
#                     new_format_fact.append(relation+str(i))
#                 # new_format_fact.pop()
#                 new_format_fact[1] = new_format_fact[1][:-1]
#                 fout.write(" ".join(new_format_fact)+"\n")

# for set in ["train","valid","test"]:
#     in_dataset_path = "data/Wiki-whole/"
#     out_dataset_path = "data/JF17K4!=0/"
#     in_path = in_dataset_path+set+".txt"
#     out_path = out_dataset_path+set+".txt"
#     with open(in_path) as f, open(out_path, "w") as fout:
#         lines = f.readlines()
#         for line in lines:
#             fact = line.split()
#             # print(fact)
#             length = len(fact)
#             if length==5:
#                 relation = fact[0]
#                 entities = fact[1:]
#                 new_format_fact = []
#                 for i in range(len(entities)):
#                     new_format_fact.append(entities[i])
#                     new_format_fact.append(relation+str(i))
#                 # new_format_fact.pop()
#                 new_format_fact[1] = new_format_fact[1][:-1]
#                 fout.write(" ".join(new_format_fact)+"\n")

# for set in ["train","valid","test"]:
#     in_dataset_path = "data/JF17K!=0/"
#     out_dataset_path = "data/JF17K-2ary/"
#     in_path = in_dataset_path+set+".txt"
#     out_path = out_dataset_path+set
#     with open(in_path) as f, open(out_path, "w") as fout:
#         lines = f.readlines()
#         count = 0
#         for line in lines:
#             fact = line.split()
#             # print(fact)
#             length = len(fact)
#             if length == 4:
#                 fout.write("\t".join(fact[:-1])+"\n")
#                 count+=1
#         print(count)

# a = torch.randn([100,128])
# b = torch.randn([100,128])
# cos = torch.cosine_similarity(a, b)
# print(cos.shape)

# a = [100,200,200]
# del a[2]
# print(a)

# import numpy as np
# from sklearn.manifold import TSNE
# X = np.array([[0, 0, 0], [0, 0.1, 0.1], [0.1, 0, 0.1], [0.1, 0.1, 0.1]])
# X_embedded = TSNE(n_components=2).fit_transform(X)
# print(X_embedded)

a = [1,2,3]
print(a[1:])
