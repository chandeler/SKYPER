import pickle
import copy

class Data:

    def __init__(self, data_dir="data/WN18RR/"):
        self.train_data = self.load_data(data_dir, "train")
        self.valid_data = self.load_data(data_dir, "valid")
        self.test_data = self.load_data(data_dir, "test")
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]
        self.primary_entities = self.get_primary_entities(self.data)
        self.affilated_entities = self.get_affiliated_entities(self.data)
        

    def load_data(self, data_dir, data_type="train"):
        with open("%s%s.txt" % (data_dir, data_type), "r", encoding="utf-8") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            # if len(data[0])>3:
            #     data += [[d[2], str(d[1]) + "_reverse", d[0]] + [item for item in d[3:]] for d in data]
            # else:
            #     data += [[d[2], d[1]+"_reverse", d[0]] for d in data]
        return data

    def get_relations(self, data):
        # 1,3,5...表示的是关系
        relations = sorted(list(set([d[i]  for d in data for i in range(1,len(d),2)])))
        # relations = sorted(list(set([d[1] for d in data])))
        # relations = sorted(list(set([d[1] for d in data])))
        relations.remove("o")
        return relations

    def get_entities(self, data):
        # 0,2,4,6....表示的是实体
        entities = sorted(list(set([d[i] for d in data for i in range(0,len(d),2) ])))
        # entities  =[]
        # for fact in data:
        #     for i in range(0,len(fact))
        return entities

    def get_primary_entities(self,data):
        pri_entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        # print(len(pri_entities))
        return pri_entities

    def get_affiliated_entities(self,data):
        entities = sorted(list(set([d[i]  for d in data for i in range(4, len(d), 2)])))
        # print(len(entities))
        return entities


class Data_whole_and_split:

    def __init__(self, data_dir="data/WN18RR/"):
        self.train_data = self.load_data(data_dir, "train")
        self.valid_data = self.load_data(data_dir, "valid")
        self.test_data = self.load_data(data_dir, "test")
        self.data = self.train_data + self.valid_data + self.test_data
        self.data_group_by_arity = self.load_data_group_by_arity(self.data)
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]
        self.toTest = self.get_query_candidates(self.test_data)
        # self.primary_entities = self.get_primary_entities(self.data)
        # self.affilated_entities = self.get_affiliated_entities(self.data)

    def load_data(self, data_dir, data_type="train"):
        with open("%s%s.txt" % (data_dir, data_type), "r", encoding="utf-8") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            # for item in data:
            #     item[3]="oo"

            for i  in range(len(data)):
                 # if len(data[i]) >= 4:
                #     data += [[data[i][2], str(data[i][1]) + "_reverse", data[i][0]] + [item for item in data[i][3:]]]
                # else:
                #     data += [[data[i][2], str(data[i][1])+"_reverse", data[i][0],"oo"]]
                data += [[data[i][2], str(data[i][1]) + "_reverse", data[i][0]] + [item for item in data[i][3:]]]
            # if len(data)==4:
        #     for i in range(len(data)):
       #         data += [[data[i][2], str(data[i][1]) + "_reverse", data[i][0]] + [item for item in data[i][3:]]]
            return data

    def load_data_group_by_arity(self, data):
        facts_group_by_arity = dict()

        for facts in data:
            if str(len(facts)) not in list(facts_group_by_arity.keys()):
                facts_group_by_arity[str(len(facts))] = []
                facts_group_by_arity[str(len(facts))].append(facts)
            else:
                facts_group_by_arity[str(len(facts))].append(facts)
        data_group_by_arity_list = list(facts_group_by_arity.values())
        return data_group_by_arity_list

    def get_max_arity(self):
        data_group_by_arity = self.load_data_group_by_arity(self.data)
        max = len(data_group_by_arity[0][0])
        for arity_data in data_group_by_arity:
            if len(arity_data[0])>max:
                max = len(arity_data[0])
        max = int(max/2)
        return max



    def get_relations(self, data):
        # 1,3,5...表示的是关系
        relations = sorted(list(set([d[i] for d in data for i in range(1, len(d), 2)])))
        # relations.remove("o")
        # relations = sorted(list(set([d[1] for d in data])))
        # relations = sorted(list(set([d[1] for d in data])))


        return relations

    def get_entities(self, data):
        # 0,2,4,6....表示的是实体
        entities = sorted(list(set([d[i] for d in data for i in range(0, len(d), 2)])))
        # entities  =[]
        # for fact in data:
        #     for i in range(0,len(fact))
        return entities
    
    def get_query_candidates(self,testdata):
        strs = 'abc-ab-queryid-1-12345'
        str_list = strs.split("-")
        # print(''.join(str_list[2:-1]))
        query_candidates = {}
        for i,test in enumerate(testdata):
            # if i <(len(testdata)//2):
            if i <(len(testdata)):
                key = test[0]
                if key not in list(query_candidates.keys()):
                    query_candidates[key]=[]
                
        for entity in self.entities:
            if 'candidate' in entity:
                candidate_query = entity.split("-")[2:-1]
                if '1071' not in candidate_query and "5180" not in candidate_query and "991" not in candidate_query and "743" not in candidate_query and "3859" not in candidate_query:
                    query = "".join(candidate_query)
                    # print(query,entity)
                else:
                    query = "-".join(candidate_query)
                    # print(query,entity)

                query_candidates[query].append(entity) if query in query_candidates else None

        return query_candidates


    def get_primary_entities(self, data):
        pri_entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        print(len(pri_entities))
        return pri_entities

    def get_affiliated_entities(self, data):
        entities = sorted(list(set([d[i] for d in data for i in range(4, len(d), 2)])))
        print(len(entities))
        return entities

if __name__ == '__main__':

    data = Data_whole_and_split(data_dir="data/LeCard-Event/")
    
    adict = data.get_query_candidates(data.test_data)
    
    
    
    
    # with open("data/JF17K-small/valid.txt", 'w') as fout:
    #
    #     with open("data/JF17K-small/valid-1.txt",'r') as fin:
    #
    #
    #         lines = fin.readlines()
    #
    #         count = 0
    #         for line in lines:
    #
    #             data = line.replace("\n",'').split(" ")
    #             data [3] = data[1]
    #             line_length = len(data)
    #             fout.write(" ".join(data)+"\n")
    #
    #             if line_length == 5:
    #                 count += 1
    #                 print(line)

    # with open("data/WikiPeople!=0/WikiPeople!=0.bin", 'rb') as fin:
    #     info = pickle.load(fin)
    #     train = info["train_facts"]
    #     valid = info["valid_facts"]
    #     test = info["test_facts"]
    #     # attr_val = info["attr_val"]
    #     # rel_head = info["rel_head"]
    #     # rel_tail = info["rel_tail"]
    #     entities_indexes = info["entities_indexes"]
    #     relations_indexes = info["relations_indexes"]
    #     binary_dict = test[1]
    #     triples = list(binary_dict.keys())
    #
    # print("helle")


    # with open("data/JF17K-small/test-1.txt",'w') as fout:
    #     for triple in triples:
    #         count = 0
    #         for element in triple:
    #             count += 1
    #             if count != len(triple):
    #                 fout.write(str(element)+ " ")
    #             else:
    #                 fout.write(str(element)+ "\n")


