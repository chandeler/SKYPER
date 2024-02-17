import numpy as np
from utils import *
import torch


class MuRP(torch.nn.Module):
    def __init__(self, d, dim, device):
        super(MuRP, self).__init__()
        self.dim = dim
        self.Eh = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.Eh.weight.data = (0.001 * torch.randn((len(d.entities), dim), dtype=torch.float32,
                                                   device=device))
        self.rvh = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rvh.weight.data = (0.001 * torch.randn((len(d.relations), dim), dtype=torch.float32,
                                                    device=device))
        self.weight_for_head = torch.nn.Parameter(
            torch.tensor(np.random.uniform(-1, 1, (len(d.relations), dim)), dtype=torch.float32, requires_grad=True,
                         device=device))

        self.bs = torch.nn.Parameter(
            torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=device))
        self.bo = torch.nn.Parameter(
            torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=device))
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, data):
        data.pop()
        u_idx = data[0]
        r_idx = data[1]
        v_idx = data[2]
        i_to_corrupt = data[-1]

        u = self.Eh.weight[u_idx]
        v = self.Eh.weight[v_idx]
        rvh = self.rvh.weight[r_idx]

        # torch.where 第一个参数表述条件，第二个参数表示符合条件时设置的值，第三个为不符合条件设置的值
        u = norm_within_one(u)
        v = norm_within_one(v)
        rvh = norm_within_one(rvh)

        weight = self.weight_for_head[r_idx]
        # weight_for_t = self.weight_for_tail[r_idx]
        if len(data)-1  == 4:

            head = norm_within_one(p_exp_map(weight * p_log_map(u)))
            tail = norm_within_one(p_sum(v, rvh))


        elif len(data)-1  == 6:

            # attr_idx = data[3]
            attr_val_idx = data[4]
            # attr = self.rvh.weight[attr_idx]
            attr_val = self.Eh.weight[attr_val_idx]
            # attr = norm_within_one(attr)
            attr_val = norm_within_one(attr_val)

            head = norm_within_one(p_exp_map(weight * (p_log_map(u) + p_log_map(attr_val))))
            tail = norm_within_one(p_exp_map(p_log_map(v) + p_log_map(attr_val)))
            tail = norm_within_one(p_sum(tail, rvh))

        else:
            attr_val_idx = data[4]
            cat = self.Eh.weight[attr_val_idx]
            cat = norm_within_one(cat)
            cat = p_log_map(cat)
            # attr_bias = self.ba[2,attr_val_idx]

            for i in range(6, len(data), 2):
                attr_inx = data[i]
                attr_emb = self.Eh.weight[attr_inx]
                attr_emb = norm_within_one(attr_emb)
                attr_emb = p_log_map(attr_emb)
                cat = torch.min(cat, attr_emb)

            head = norm_within_one(p_exp_map(weight * (p_log_map(u) + cat)))
            tail = norm_within_one(p_exp_map(p_log_map(v) + cat))
            tail = norm_within_one(p_sum(tail, rvh))

        dist = calculate_dist(head, tail)
        # if i_to_corrupt == False:
        #     dist[:,1:] = 1.5*dist[:,1:]
        # score = -dist + torch.abs(self.bs[u_idx]) + torch.abs(self.bo[v_idx])
        score = -dist + self.bs[u_idx] + self.bo[v_idx]
        # score = -dist + self.bs[u_idx] + self.bo[v_idx]+self.ba[attr_val_idx]

        return score


class Centroid(torch.nn.Module):
    def __init__(self, d, dim, device):
        super(Centroid, self).__init__()
        self.dim = dim
        self.max_ary = d.get_max_arity()
        self.Eh = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.Eh.weight.data = (0.005 * torch.randn((len(d.entities), dim), dtype=torch.float32,
                                                      device=device))
        self.rvh = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rvh.weight.data = (0.005* torch.randn((len(d.relations), dim), dtype=torch.float32,
                                                       device=device))
        self.weight_for_head = torch.nn.Parameter(
                torch.tensor(np.random.uniform(-1, 1, (len(d.relations), dim)), dtype=torch.float32, requires_grad=True,
                             device=device))
            # self.Plist = torch.nn.ParameterList(
            #     [torch.nn.Parameter((torch.rand(K, arity, self.n_parts, requires_grad=True)).to(device))
            #      for arity in range(2, self.max_ary + 1)])
        # self.bs = torch.nn.Parameter(
        #             torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device=device)
        #         )
        #
        # self.bo = torch.nn.Parameter(
        #             torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device=device)
        #         )
        self.bias = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=device)
                ) for _ in range(self.max_ary)
        ]
        )

        # self.b1 = torch.nn.Parameter(
        #     torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=device))
        # self.b2 = torch.nn.Parameter(
        #     torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=device))
        # self.b3 = torch.nn.Parameter(
        #     torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=device))
        # self.b4 = torch.nn.Parameter(
        #     torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=device))
        # self.b5 = torch.nn.Parameter(
        #     torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=device))

        # self.bias = torch.zeros(d.get_max_arity(),len(d.entities), dtype=torch.float32, requires_grad=True, device=device),


        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, data):


        data.pop()
        rand = [0]
        corrupt_domain = data[-1]
        rand.insert(0,corrupt_domain)
        # corrupt_next = int(corrupt_domain / 2) + 1
        entity_domain = [i for i in range(0, len(data) - 1, 2)]

        # if corrupt_domain != entity_domain[-1]:
        #     rand.append(entity_domain[corrupt_next])
        # else:
        #     rand.append(entity_domain[0])

        # rest = list(set(entity_domain) - set(rand))

        real_head_index = data[0]
        real_tail_index = data[2]
        real_head_embedding_1 = self.Eh.weight[real_head_index]
        real_tail_embedding_1 = self.Eh.weight[real_tail_index]
        rvh_1 = self.rvh.weight[data[1]]
        weight_1 = self.weight_for_head[data[1]]

        # rvh_1 = self.weight_for_head[data[1]]
        # weight_1 = self.weight_for_head[data[3]]
        real_tail_embedding = norm_within_one(real_tail_embedding_1)
        real_head_embedding = norm_within_one(real_head_embedding_1)
        rvh = norm_within_one(rvh_1)





        # weight_1 = self.rvh.weight[data[1]]


        # head = norm_within_one(p_sum(norm_within_one(real_head_embedding), weight_1))
        h_e = p_log_map(real_head_embedding)
        h_W = h_e*weight_1
        h_m = p_exp_map(h_W)
        head = norm_within_one(h_m)



        # head = norm_within_one(p_exp_map(p_log_map(real_head_embedding)*rvh_1))


        tail_0 = p_sum(real_tail_embedding, rvh)
        tail = norm_within_one(tail_0)

        # tail = norm_within_one(p_exp_map(p_log_map(real_tail_embedding) * weight_1))
        midpoint = cal_midpoint(head,tail)
        # Dist_HT = 3*calculate_dist(midpoint, tail)

        Dist_HT = calculate_dist(head,tail)

        # HR_idx = data[rand[0] + 1]
        # TR_idx = data[rand[1] + 1]


        # weight = norm_within_one(self.rvh.weight[HR_idx])
        # weight = self.weight_for_head[HR_idx]
        # rvh = self.rvh.weight[TR_idx]


        # rand_head_inx = data[rand[0]]
        # rand_tail_idx = data[rand[1]]
        # rand_head_embedding = self.Eh.weight[rand_head_inx]
        # rand_tail_embedding = self.Eh.weight[rand_tail_idx]
        # head_1 = norm_within_one(p_sum(norm_within_one(rand_head_embedding), weight))
        # head_1 = norm_within_one(p_exp_map(weight * p_log_map(rand_head_embedding)))
        # tail_1 = norm_within_one(p_sum(norm_within_one(rand_tail_embedding), rvh))
        #
        # len_HT = calculate_dist(head_1, tail_1)

        # entity_index = [data[i] for i in entity_domain]
        local_bias = self.bias[0][real_head_index]+self.bias[1][real_tail_index]
        # local_bias = self.bs[real_head_index]+self.bo[real_tail_index]
        len_ht = -Dist_HT+local_bias
        #         # for i in range(len(entity_index)):
        #     bias += self.bias[i][entity_index[i]]

        if len(data) - 1 == 4:
            # score = -len_HT+self.bias[0][int(rand[0]/2),rand_head_inx]+self.bias[0][int(rand[1]/2),rand_tail_idx]
            score = len_ht
            # score = -len_HT
        else:

            """计算corrupt的节点到质心测地线长度+AB距离"""
            fact_entities_embedding = [norm_within_one(self.Eh.weight[data[2*i]]) for i in range(2,len(entity_domain))]
            fact_relation_embedding = [norm_within_one(self.rvh.weight[data[2*i + 1]]) for i in range(2,len(entity_domain))]



            new_entities_embedding = [norm_within_one(p_sum(fact_entities_embedding[i], fact_relation_embedding[i])) for
                                      i in range(len(fact_relation_embedding))]

            # fact_relation_embedding = [norm_within_one(self.weight_for_head[data[2 * i + 1]]) for i in
            #                            range(2, len(entity_domain))]
            # new_entities_embedding = [norm_within_one(p_exp_map(p_log_map(fact_entities_embedding[i])*fact_relation_embedding[i])) for
            #                           i in range(len(fact_relation_embedding))]

            new_entities_embedding.insert(0, tail)
            new_entities_embedding.insert(0, head)

            """计算质心"""
            corrupt_entity = new_entities_embedding[int(corrupt_domain / 2)]
            del new_entities_embedding[int(corrupt_domain / 2)]
            centroid = cal_centroid_multi(new_entities_embedding)

            corrupt_to_centroid = calculate_dist(corrupt_entity, centroid)
            #
            global_bias = 0
            for i in range(len(entity_domain)):
                global_bias += self.bias[i][data[2*i]]
            # corrupt_to_centroid = cal_perimeter_n(new_entities_embedding)

            global_score = -corrupt_to_centroid+global_bias

            # centroid_midpoint_distance = -calculate_dist(midpoint,centroid)+global_bias
            
            # score = global_score + 0.25*len_ht+ 0.25*centroid_midpoint_distance

            # score = global_score

            score = global_score + 0.1*len_ht

            # score = global_score + len_ht

            # score = global_score - 0.5*len_ht
        return score



class LeCentroid(torch.nn.Module):
    def __init__(self, d, dim, device):
        super(LeCentroid, self).__init__()
        self.dim = dim
        self.max_ary = d.get_max_arity()
        self.Eh = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.Eh.weight.data = (0.001 * torch.randn((len(d.entities), dim), dtype=torch.float32,
                                                      device=device))
        self.rvh = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rvh.weight.data = (0.001* torch.randn((len(d.relations), dim), dtype=torch.float32,
                                                       device=device))
        self.weight_for_head = torch.nn.Parameter(
                torch.tensor(np.random.uniform(-1, 1, (len(d.relations), dim)), dtype=torch.float32, requires_grad=True,
                             device=device))
           
        # self.bs = torch.nn.Parameter(
        #             torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device=device)
        #         )
        #
        
        self.bias = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=device)
                ) for _ in range(self.max_ary)
        ]
        )

    def forward(self, data, scr_data=None):

        def cal_scr_score():
            if scr_data is not None:
                scr_emb = self.Eh.weight[scr_data]
                bias = self.bias[1][scr_data].sum(dim=1)
                return calculate_dist(scr_emb[:, 0, :], scr_emb[:, 1, :]) + bias
            else:
                return None

        scr_score = cal_scr_score()

        to_test = data[-1]
        data.pop()
        corrupt_domain = data[-1]
        # rand.insert(0,corrupt_domain)
        # corrupt_next = int(corrupt_domain / 2) + 1
        entity_domain = [i for i in range(0, len(data) - 1, 2)]


        real_head_index = data[0]
        real_tail_index = data[2]
        
        real_head_embedding_1 = self.Eh.weight[real_head_index]
        real_tail_embedding_1 = self.Eh.weight[real_tail_index]
        
        # rvh_1 = self.rvh.weight[data[1]]
        # weight_1 = self.weight_for_head[data[1]]
        # weight_for_head_cat = 

        fact_relation_embedding_1 = [norm_within_one(self.rvh.weight[data[2*i + 1]]) for i in range(2,len(entity_domain))]
        fact_relation_embedding_1.insert(0,norm_within_one(self.rvh.weight[data[1]]))
        fact_entities_embedding_1 = [norm_within_one(self.Eh.weight[data[2*i]]) for i in range(2,len(entity_domain))]
        fact_entities_embedding_1.insert(0,norm_within_one(self.Eh.weight[data[2]]))
        
        fact_weight_embedding_1 = [norm_within_one(self.weight_for_head[data[2*i + 1]]) for i in range(2,len(entity_domain))]
        fact_weight_embedding_1.insert(0,norm_within_one(self.weight_for_head[data[1]]))
        
        new_entity_embedding_1 = [norm_within_one(p_sum(fact_entities_embedding_1[i], fact_relation_embedding_1[i])) for i in range(len(fact_relation_embedding_1))]
        
        
        translated_tail = cal_centroid_multi(new_entity_embedding_1)

        rvh_1 = cat_rel(fact_relation_embedding_1)
        # weight_1 = self.weight_for_head[data[1]]

        weight_1 = cat_rel(fact_weight_embedding_1)



        # rvh_1 = self.weight_for_head[data[1]]
        # weight_1 = self.weight_for_head[data[3]]
        real_tail_embedding = norm_within_one(real_tail_embedding_1)
        real_head_embedding = norm_within_one(real_head_embedding_1)
        rvh = norm_within_one(rvh_1)

        real_tail_embedding = p_sum(real_tail_embedding, self.rvh.weight[data[1]])



        h_e = p_log_map(real_head_embedding)
        real_head_embedding_1 = p_exp_map(h_e*weight_1)
        # h_W = p_sum(h_e,rvh)

        # h_W = p_sum(real_head_embedding, rvh)
        # h_m = p_exp_map(h_W)
        # head = norm_within_one(h_m)

        head = real_head_embedding_1



        
        tail_0 = translated_tail

        tail = real_tail_embedding

        # midpoint = cal_midpoint(head,tail)

        Dist_HT = calculate_dist(head,tail_0)

        local_bias = self.bias[0][real_head_index]+self.bias[1][real_tail_index]
        len_ht = -Dist_HT+local_bias
        
        # tail_0 = p_sum(real_tail_embedding, rvh)

        if len(data) - 1 == 4 or len(data) - 1 == 3:
            score = len_ht
            return score, scr_score
        elif len(data) - 1 > 4:

            """计算corrupt的节点到质心测地线长度+AB距离"""
            # fact_entities_embedding = [norm_within_one(self.Eh.weight[data[2*i]]) for i in range(2,len(entity_domain))]
            # fact_relation_embedding = [norm_within_one(self.rvh.weight[data[2*i + 1]]) for i in range(2,len(entity_domain))]



            # new_entities_embedding = [norm_within_one(p_sum(fact_entities_embedding[i], fact_relation_embedding[i])) for
            #                           i in range(len(fact_relation_embedding))]


            # # new_entities_embedding.insert(0, tail)

            # corrupt_entity_1 = head

            # new_tail = cal_centroid_multi(new_entities_embedding)

            # head_to_new_tail = calculate_dist(corrupt_entity_1, new_tail)

            # del new_entities_embedding[0]

            # new_entities_embedding.insert(0, head)

            # new_head = cal_centroid_multi(new_entities_embedding)

            del new_entity_embedding_1[0]
            new_entity_embedding_1.insert(0,real_head_embedding_1)
            
            corrupt_entity_2 = real_tail_embedding

            new_head = cal_centroid_multi(new_entity_embedding_1)

            tail_to_new_head = calculate_dist(corrupt_entity_2, new_head)

            new_entity_embedding_1.insert(1, real_tail_embedding)
            # new_entities_embedding.insert(0, head)

            """计算质心"""
            corrupt_entity = new_entity_embedding_1[int(corrupt_domain / 2)]
            del new_entity_embedding_1[int(corrupt_domain / 2)]
            centroid = cal_centroid_multi(new_entity_embedding_1)

            corrupt_to_centroid = calculate_dist(corrupt_entity, centroid)
            #
            global_bias = 0
            for i in range(len(entity_domain)):
                global_bias += self.bias[i][data[2*i]]
            # corrupt_to_centroid = cal_perimeter_n(new_entities_embedding)

            global_score = -corrupt_to_centroid-0.2*Dist_HT-0.2*tail_to_new_head+local_bias

            # new try
            # global_score = -corrupt_to_centroid-0.5*tail_to_new_head-global_bias

            global_score = -corrupt_to_centroid-tail_to_new_head-global_bias

            # global_score = -1.5*corrupt_to_centroid-0.2*Dist_HT+global_bias



            # global_score = -corrupt_to_centroid

            # centroid_midpoint_distance = -calculate_dist(midpoint,centroid)
            
            # score = global_score + 0.25*len_ht+ 0.25*centroid_midpoint_distance

            score = global_score

            # score = global_score + 0.1*len_ht

            # score = global_score + len_ht

            # score = global_score - 0.5*len_ht
            if not to_test:
                return score, scr_score
            else:
                return -corrupt_to_centroid+local_bias, scr_score

class Cos_Centroid(torch.nn.Module):
    def __init__(self, d, dim, device):
        super(Centroid, self).__init__()
        self.dim = dim
        self.max_ary = d.get_max_arity()
        self.Eh = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.Eh.weight.data = (0.005 * torch.randn((len(d.entities), dim), dtype=torch.float32,
                                                      device=device))
        self.rvh = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rvh.weight.data = (0.005* torch.randn((len(d.relations), dim), dtype=torch.float32,
                                                       device=device))
        self.weight_for_head = torch.nn.Parameter(
                torch.tensor(np.random.uniform(-1, 1, (len(d.relations), dim)), dtype=torch.float32, requires_grad=True,
                             device=device))
            # self.Plist = torch.nn.ParameterList(
            #     [torch.nn.Parameter((torch.rand(K, arity, self.n_parts, requires_grad=True)).to(device))
            #      for arity in range(2, self.max_ary + 1)])
        # self.bs = torch.nn.Parameter(
        #             torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device=device)
        #         )
        #
        # self.bo = torch.nn.Parameter(
        #             torch.zeros(len(d.entities), dtype=torch.double, requires_grad=True, device=device)
        #         )
        self.bias = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=device)
                ) for _ in range(self.max_ary)
        ]
        )

        # self.b1 = torch.nn.Parameter(
        #     torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=device))
        # self.b2 = torch.nn.Parameter(
        #     torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=device))
        # self.b3 = torch.nn.Parameter(
        #     torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=device))
        # self.b4 = torch.nn.Parameter(
        #     torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=device))
        # self.b5 = torch.nn.Parameter(
        #     torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=device))

        # self.bias = torch.zeros(d.get_max_arity(),len(d.entities), dtype=torch.float32, requires_grad=True, device=device),


        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, data):


        data.pop()
        rand = [0]
        corrupt_domain = data[-1]
        rand.insert(0,corrupt_domain)
        # corrupt_next = int(corrupt_domain / 2) + 1
        entity_domain = [i for i in range(0, len(data) - 1, 2)]

        # if corrupt_domain != entity_domain[-1]:
        #     rand.append(entity_domain[corrupt_next])
        # else:
        #     rand.append(entity_domain[0])

        # rest = list(set(entity_domain) - set(rand))

        real_head_index = data[0]
        real_tail_index = data[2]
        real_head_embedding_1 = self.Eh.weight[real_head_index]
        real_tail_embedding_1 = self.Eh.weight[real_tail_index]
        rvh_1 = self.rvh.weight[data[1]]
        weight_1 = self.weight_for_head[data[1]]

        # rvh_1 = self.weight_for_head[data[1]]
        # weight_1 = self.weight_for_head[data[3]]
        real_tail_embedding = norm_within_one(real_tail_embedding_1)
        real_head_embedding = norm_within_one(real_head_embedding_1)
        rvh = norm_within_one(rvh_1)





        # weight_1 = self.rvh.weight[data[1]]


        # head = norm_within_one(p_sum(norm_within_one(real_head_embedding), weight_1))
        h_e = p_log_map(real_head_embedding)
        h_W = h_e*weight_1
        h_m = p_exp_map(h_W)
        head = norm_within_one(h_m)



        # head = norm_within_one(p_exp_map(p_log_map(real_head_embedding)*rvh_1))


        tail_0 = p_sum(real_tail_embedding, rvh)
        tail = norm_within_one(tail_0)

        # tail = norm_within_one(p_exp_map(p_log_map(real_tail_embedding) * weight_1))
        midpoint = cal_midpoint(head,tail)
        # Dist_HT = 3*calculate_dist(midpoint, tail)

        Dist_HT = calculate_dist(head,tail)

        # HR_idx = data[rand[0] + 1]
        # TR_idx = data[rand[1] + 1]


        # weight = norm_within_one(self.rvh.weight[HR_idx])
        # weight = self.weight_for_head[HR_idx]
        # rvh = self.rvh.weight[TR_idx]


        # rand_head_inx = data[rand[0]]
        # rand_tail_idx = data[rand[1]]
        # rand_head_embedding = self.Eh.weight[rand_head_inx]
        # rand_tail_embedding = self.Eh.weight[rand_tail_idx]
        # head_1 = norm_within_one(p_sum(norm_within_one(rand_head_embedding), weight))
        # head_1 = norm_within_one(p_exp_map(weight * p_log_map(rand_head_embedding)))
        # tail_1 = norm_within_one(p_sum(norm_within_one(rand_tail_embedding), rvh))
        #
        # len_HT = calculate_dist(head_1, tail_1)

        # entity_index = [data[i] for i in entity_domain]
        local_bias = self.bias[0][real_head_index]+self.bias[1][real_tail_index]
        # local_bias = self.bs[real_head_index]+self.bo[real_tail_index]
        len_ht = -Dist_HT+local_bias
        #         # for i in range(len(entity_index)):
        #     bias += self.bias[i][entity_index[i]]

        if len(data) - 1 == 4:
            # score = -len_HT+self.bias[0][int(rand[0]/2),rand_head_inx]+self.bias[0][int(rand[1]/2),rand_tail_idx]
            score = len_ht
            # score = -len_HT
        else:

            """计算corrupt的节点到质心测地线长度+AB距离"""
            fact_entities_embedding = [norm_within_one(self.Eh.weight[data[2*i]]) for i in range(2,len(entity_domain))]
            fact_relation_embedding = [norm_within_one(self.rvh.weight[data[2*i + 1]]) for i in range(2,len(entity_domain))]



            new_entities_embedding = [norm_within_one(p_sum(fact_entities_embedding[i], fact_relation_embedding[i])) for
                                      i in range(len(fact_relation_embedding))]

            # fact_relation_embedding = [norm_within_one(self.weight_for_head[data[2 * i + 1]]) for i in
            #                            range(2, len(entity_domain))]
            # new_entities_embedding = [norm_within_one(p_exp_map(p_log_map(fact_entities_embedding[i])*fact_relation_embedding[i])) for
            #                           i in range(len(fact_relation_embedding))]

            new_entities_embedding.insert(0, tail)
            new_entities_embedding.insert(0, head)

            """计算质心"""
            corrupt_entity = new_entities_embedding[int(corrupt_domain / 2)]
            del new_entities_embedding[int(corrupt_domain / 2)]
            centroid = cal_centroid_multi(new_entities_embedding)

            corrupt_to_centroid = calculate_dist(corrupt_entity, centroid)
            #
            global_bias = 0
            for i in range(len(entity_domain)):
                global_bias += self.bias[i][data[2*i]]
            # corrupt_to_centroid = cal_perimeter_n(new_entities_embedding)

            global_score = -corrupt_to_centroid+global_bias

            centroid_midpoint_distance = -calculate_dist(midpoint,centroid)
            score = global_score + 0.25*len_ht+ 0.25*centroid_midpoint_distance

            # score = global_score + len_ht
        return score

class WPolygonE(torch.nn.Module):
    def __init__(self, d, dim, device):
        super(WPolygonE, self).__init__()
        self.dim = dim
        self.max_ary = d.get_max_arity()
        self.Eh = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.Eh.weight.data = (0.005 * torch.randn((len(d.entities), dim), dtype=torch.float32,
                                                      device=device))
        self.rvh = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rvh.weight.data = (0.005* torch.randn((len(d.relations), dim), dtype=torch.float32,
                                                       device=device))
        self.weight_for_head = torch.nn.Parameter(
                torch.tensor(np.random.uniform(-1, 1, (len(d.relations), dim)), dtype=torch.float32, requires_grad=True,
                             device=device))

        self.bias = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=device)
                ) for _ in range(self.max_ary)
        ]
        )


        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, data):


        data.pop()
        rand = [0]
        corrupt_domain = data[-1]
        rand.insert(0,corrupt_domain)

        entity_domain = [i for i in range(0, len(data) - 1, 2)]
        real_head_index = data[0]
        real_tail_index = data[2]
        real_head_embedding_1 = self.Eh.weight[real_head_index]
        real_tail_embedding_1 = self.Eh.weight[real_tail_index]
        rvh_1 = self.rvh.weight[data[1]]
        weight_1 = self.weight_for_head[data[1]]
        real_tail_embedding = norm_within_one(real_tail_embedding_1)
        real_head_embedding = norm_within_one(real_head_embedding_1)
        rvh = norm_within_one(rvh_1)

        h_e = p_log_map(real_head_embedding)
        h_W = h_e*weight_1
        h_m = p_exp_map(h_W)
        head = norm_within_one(h_m)


        tail_0 = p_sum(real_tail_embedding, rvh)
        tail = norm_within_one(tail_0)


        midpoint = cal_midpoint(head,tail)


        Dist_HT = calculate_dist(head,tail)


        local_bias = self.bias[0][real_head_index]+self.bias[1][real_tail_index]

        len_ht = -Dist_HT+local_bias


        if len(data) - 1 == 4:

            score = len_ht

        else:

            """计算corrupt的节点到质心测地线长度+AB距离"""
            fact_entities_embedding = [norm_within_one(self.Eh.weight[data[2*i]]) for i in range(2,len(entity_domain))]
            fact_relation_embedding = [norm_within_one(self.rvh.weight[data[2*i + 1]]) for i in range(2,len(entity_domain))]

            new_entities_embedding = [norm_within_one(p_sum(fact_entities_embedding[i], fact_relation_embedding[i])) for
                                      i in range(len(fact_relation_embedding))]



            new_entities_embedding.insert(0, tail)
            new_entities_embedding.insert(0, head)

            """计算质心"""
            corrupt_entity = new_entities_embedding[int(corrupt_domain / 2)]
            del new_entities_embedding[int(corrupt_domain / 2)]
            centroid = cal_weighted_centroid_multi(new_entities_embedding)

            corrupt_to_centroid = calculate_dist(corrupt_entity, centroid)

            global_bias = 0
            for i in range(len(entity_domain)):
                global_bias += self.bias[i][data[2*i]]


            global_score = -corrupt_to_centroid+global_bias

            centroid_midpoint_distance = -calculate_dist(midpoint,centroid)
            score = 1*global_score + len_ht+ 0.25*centroid_midpoint_distance

        return score



class all_Centroid(torch.nn.Module):
    def __init__(self, d, dim, device):
        super(all_Centroid, self).__init__()
        self.dim = dim
        self.max_ary = d.get_max_arity()
        self.Eh = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.Eh.weight.data = (0.001 * torch.randn((len(d.entities), dim), dtype=torch.float32,
                                                      device=device))
        self.rvh = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rvh.weight.data = (0.001 * torch.randn((len(d.relations), dim), dtype=torch.float32,
                                                       device=device))
        self.weight_for_head = torch.nn.Parameter(
                torch.tensor(np.random.uniform(-1, 1, (len(d.relations), dim)), dtype=torch.float32, requires_grad=True,
                             device=device))


        self.bias = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=device)
                ) for _ in range(self.max_ary)
        ]
        )
        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, data):

        do_test = data[-1]
        data.pop()
        rand = [0]
        corrupt_domain = data[-1]
        rand.insert(0, corrupt_domain)

        entity_domain = [i for i in range(0, len(data) - 1, 2)]

        real_head_index = data[0]
        real_tail_index = data[2]
        real_head_embedding_1 = self.Eh.weight[real_head_index]
        real_tail_embedding_1 = self.Eh.weight[real_tail_index]
        rvh_1 = self.rvh.weight[data[1]]
        weight_1 = self.weight_for_head[data[1]]

        real_tail_embedding = norm_within_one(real_tail_embedding_1)
        real_head_embedding = norm_within_one(real_head_embedding_1)
        rvh = norm_within_one(rvh_1)

        h_e = p_log_map(real_head_embedding)
        h_W = h_e * weight_1
        h_m = p_exp_map(h_W)
        head = norm_within_one(h_m)


        tail_0 = p_sum(real_tail_embedding, rvh)
        tail = norm_within_one(tail_0)


        Dist_HT = calculate_dist(head, tail)


        local_bias = self.bias[0][real_head_index] + self.bias[1][real_tail_index]
        # local_bias = self.bs[real_head_index]+self.bo[real_tail_index]
        len_ht = -Dist_HT + local_bias


        if len(data) - 1 == 4:

            score = len_ht

        else:

            """计算corrupt的节点到质心测地线长度+AB距离"""
            fact_entities_embedding = [norm_within_one(self.Eh.weight[data[2 * i]]) for i in
                                       range(2, len(entity_domain))]
            fact_relation_embedding = [norm_within_one(self.rvh.weight[data[2 * i + 1]]) for i in
                                       range(2, len(entity_domain))]

            new_entities_embedding = [norm_within_one(p_sum(fact_entities_embedding[i], fact_relation_embedding[i])) for
                                      i in range(len(fact_relation_embedding))]
            new_entities_embedding.insert(0, tail)
            new_entities_embedding.insert(0, head)

            """计算质心"""
            corrupt_entity = new_entities_embedding[int(corrupt_domain / 2)]


            if do_test== False:
                new_entities_embedding_all = [new_entities_embedding[i][:, 0, :] for i in
                                                  range(len(new_entities_embedding))]
                centroid = cal_centroid_multi(new_entities_embedding_all)

                shapes = corrupt_entity.shape
                centroid_repeat = centroid.repeat(1,shapes[1]).reshape(shapes[0],shapes[1],shapes[2])

                corrupt_to_centroid = calculate_dist(corrupt_entity, centroid_repeat)

            elif do_test == True:
                centroid = cal_centroid_multi(new_entities_embedding)
                corrupt_to_centroid = calculate_dist(corrupt_entity, centroid)

            #
            global_bias = 0
            for i in range(2, len(entity_domain)):
                global_bias += self.bias[i][data[2 * i]]
            # corrupt_to_centroid = cal_perimeter_n(new_entities_embedding)

            global_score = -corrupt_to_centroid + global_bias

            score = global_score + len_ht
        return score

class E_centroid(torch.nn.Module):
    def __init__(self, d, dim, device):
        super(E_centroid, self).__init__()
        self.dim = dim
        self.max_ary = d.get_max_arity()
        self.Eh = torch.nn.Embedding(len(d.entities), dim, padding_idx=0)
        self.Eh.weight.data = (0.001 * torch.randn((len(d.entities), dim), dtype=torch.float32,
                                                   device=device))
        self.rvh = torch.nn.Embedding(len(d.relations), dim, padding_idx=0)
        self.rvh.weight.data = (0.001 * torch.randn((len(d.relations), dim), dtype=torch.float32,
                                                    device=device))
        self.weight_for_head = torch.nn.Parameter(
            torch.tensor(np.random.uniform(-1, 1, (len(d.relations), dim)), dtype=torch.float32, requires_grad=True,
                         device=device))

        self.bias = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.zeros(len(d.entities), dtype=torch.float32, requires_grad=True, device=device)
                ) for _ in range(self.max_ary)
            ]
        )



        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, data):

        data.pop()
        rand = [0]
        corrupt_domain = data[-1]
        rand.insert(0, corrupt_domain)
        entity_domain = [i for i in range(0, len(data) - 1, 2)]


        real_head_index = data[0]
        real_tail_index = data[2]
        real_head_embedding_1 = self.Eh.weight[real_head_index]
        real_tail_embedding_1 = self.Eh.weight[real_tail_index]
        rvh_1 = self.rvh.weight[data[1]]
        weight_1 = self.weight_for_head[data[1]]
        weight_1 = self.rvh.weight[data[3]]

        real_tail_embedding = real_tail_embedding_1
        real_head_embedding = real_head_embedding_1
        rvh = rvh_1



        head = real_head_embedding




        tail = real_tail_embedding+rvh


        Dist_HT = torch.norm(head-tail,dim=-1)**2

        


        local_bias = self.bias[0][real_head_index] + self.bias[1][real_tail_index]
        # local_bias = 0

        len_ht = -Dist_HT + local_bias


        if len(data) - 1 == 4:

            score = len_ht

        else:

            """计算corrupt的节点到质心测地线长度+AB距离"""
            fact_entities_embedding = [self.Eh.weight[data[2 * i]] for i in
                                       range(2, len(entity_domain))]
            fact_relation_embedding = [self.rvh.weight[data[2 * i + 1]] for i in
                                       range(2, len(entity_domain))]

            new_entities_embedding = [fact_entities_embedding[i]+fact_relation_embedding[i] for
                                      i in range(len(fact_relation_embedding))]
            new_entities_embedding.insert(0, tail)
            new_entities_embedding.insert(0, head)

            """计算质心"""
            corrupt_entity = new_entities_embedding[int(corrupt_domain / 2)]
            del new_entities_embedding[int(corrupt_domain / 2)]
            centroid = cal_centroid_Eu(new_entities_embedding)

            corrupt_to_centroid = torch.norm(corrupt_entity-centroid,dim=-1)**2
            #
            global_bias = 0
            for i in range(2, len(entity_domain)):
                global_bias += self.bias[i][data[2 * i]]


            global_score = -corrupt_to_centroid + global_bias

            score = global_score + 0.1*len_ht
        return score





  

