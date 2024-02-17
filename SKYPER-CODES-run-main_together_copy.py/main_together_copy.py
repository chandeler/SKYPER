import numpy as np
import torch as torch
import os
import copy
import time
import json
from datetime import datetime
from collections import defaultdict
from load_data import *
from model import *
from rsgd import *
from rsgd import *
import argparse
from log import logger
import torch.optim as optim
from utils import *
from util.gen_metric import Metric
from Contrasive import ContrasiveLoss

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


class Experiment:

    def __init__(
            self, learning_rate=50, dim=40, nneg=50, model="poincare", num_iterations=600, batch_size=128,
            cuda=False, device_ids="0", do_test=False, test_batch_size=8, use_scr=False
    ):

        self.model = model
        self.device = "cuda" if cuda else "cpu"
        self.device_ids = device_ids
        self.n_gpu = len(self.device_ids.split(",")) if cuda else 0
        self.learning_rate = learning_rate
        self.dim = dim
        self.nneg = nneg
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.cuda = cuda
        self.do_test = do_test
        self.test_batch_size = test_batch_size
        self.use_scr = use_scr

    """
    这里面做的事情是id2index 把一个n-ary fact 如 Q1138 p131 Q145 Q178 变成 360 25 177 69
    """

    def get_data_idxs(self, data):
        data_idxs = []
        for d in data:
            dataid_2_relationssid = []
            for i in range(len(d)):
                if i % 2 == 1:
                    dataid_2_relationssid.append(self.relation_idxs[d[i]])
                else:
                    dataid_2_relationssid.append(self.entity2idxs[d[i]])
            data_idxs.append(tuple(dataid_2_relationssid))

        return data_idxs
    
    def get_to_test(self, testdatadict):
        adict = {}
        for query in list(testdatadict.keys()):
            adict[self.entity2idxs[query]]=[ self.entity2idxs[candidate] for candidate in testdatadict[query]]
        return adict
    
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        ht_vocab = defaultdict(list)
        for fact in data:
            if len(fact) > 3:
                rel = fact[1]
                raw_fact =  [fact[i] for i in range(0, len(fact), 2)]
                raw_fact.insert(0,rel)
                for filter_pos in range(0, len(fact), 2):
                    triple1 = copy.deepcopy(list(raw_fact))
                    triple1[int(filter_pos/2)+1]=(int(filter_pos/2)+1)*111111
                    er_vocab[tuple(triple1)].append(fact[filter_pos])

                    # remove self-loop
                    for i in range(1,len(raw_fact)):
                         er_vocab[tuple(triple1)].append(raw_fact[i])

             #for binary fact test
            er_vocab[(fact[0], fact[1])].append(fact[2])

            er_vocab[(fact[1], fact[2])].append(fact[0])

        return er_vocab, ht_vocab

    def get_test_er_vocab(self,testdata,traindata):
        er_vocab = defaultdict(list)
        for fact in testdata:
            pass
    
    def find_KNN(self,model,data):
        test_data_group_by_arity = data.load_data_group_by_arity(data.test_data)
        logger.info("Starting testing...")

        # all_fact = self.get_data_idxs(data.data)
        # er_vocab, ht_vocab = self.get_er_vocab(all_fact)
        q2c = self.get_to_test(data.toTest)
        output_dict = {}

        for arity_data in test_data_group_by_arity[:]:

            hits_this_arity = []
            ranks_this_arity = []
            for p in range(10):
                hits_this_arity.append([])

            test_data_idxs = self.get_data_idxs(arity_data)
            fact_length = len(test_data_idxs[0])

            for j in range(0, len(test_data_idxs), self.test_batch_size):
                # print(j)
                # for i in (list(range(2, fact_length, 2))):
                for i in (list(range(2, 4, 2))):
                    if i > 2 and j < (len(test_data_idxs) / 2):
                        continue

                    if j + self.test_batch_size < len(test_data_idxs):
                        test_data_batch = np.array(test_data_idxs[j:j + self.test_batch_size])
                    else:
                        continue
                        test_data_batch = np.array(test_data_idxs[j:len(test_data_idxs)])

                    temp = []

                    if i == 2 and j >= (len(test_data_idxs) / 2):
                        # all_samples = np.repeat(range(len(data.entities)), test_data_batch.shape[0]).reshape(-1,
                        #                                                                               test_data_batch.shape[0]).T
                        all_samples = np.repeat(q2c[test_data_batch[0][2]], test_data_batch.shape[0]).reshape(-1,
                                                                                                      test_data_batch.shape[0]).T

                                                                                      
                    else: 
                        continue
                        all_samples = np.repeat(range(len(data.entities)), test_data_batch.shape[0]).reshape(-1,
                                                                                                       test_data_batch.shape[0]).T
                    
                    if i == 2 and j >= (len(test_data_idxs) / 2):
                        for k in range(len(test_data_idxs[0])):
                            temp.append(
                                torch.LongTensor(np.tile(np.array([test_data_batch[:,k]]).T, (1, len(q2c[test_data_batch[0][2]])))))
                    else:
                        for k in range(len(test_data_idxs[0])):
                            temp.append(
                                torch.LongTensor(np.tile(np.array([test_data_batch[:, k]]).T, (1, len(data.entities)))))
                    
                    candidates = torch.LongTensor(all_samples)
                    temp[i]=torch.LongTensor(all_samples)

                    query = temp[0]

                    e_i = test_data_batch[:, 2]
                    temp.append(i)
                    temp.append(True)

                    predictions, _ = model.forward(temp)

                    # predictions = -torch.sum(model.Eh.weight[query]*model.Eh.weight[candidates],dim=-1)
                    # dotxy = torch.squeeze(torch.sum(model.Eh.weight[query]*model.Eh.weight[candidates],dim=-1,keepdim=True),dim=2)
                    
                    # norm_x = torch.squeeze(torch.norm(model.Eh.weight[query], dim=-1,keepdim=True),dim=2)
                    # norm_y = torch.squeeze(torch.norm(model.Eh.weight[candidates], dim=-1,keepdim=True),dim=2)
                    # a = model.Eh.weight[candidates]-model.Eh.weight[query]
                    # norm_x_minus_y = torch.squeeze(torch.norm(a, dim=-1,keepdim=True),dim=2)
                    # predictions = norm_x**2+norm_y**2-norm_x_minus_y**2/(norm_x*norm_y*2)

                    # predictions_1 = torch.cosine_similarity(torch.squeeze(model.Eh.weight[query],0), torch.squeeze(model.Eh.weight[candidates],0))
                    # predictions = torch.unsqueeze(predictions_1, 0)

                    # predictions = calculate_dist(model.Eh.weight[candidates], model.Eh.weight[query])
                    # predictions = -torch.sum(model.Eh.weight[query]*model.Eh.weight[candidates],dim=-1,keepdim=True).squeeze(2)
                    # b = p_log_map(model.Eh.weight[query])
                    # predictions = torch.cosine_similarity(p_log_map(model.Eh.weight[query]), p_log_map(model.Eh.weight[candidates]))
                    # a = torch.norm(p_log_map(model.Eh.weight[query]),2,dim=1)
                    # predictions = torch.sum(p_log_map(model.Eh.weight[query])*p_log_map(model.Eh.weight[candidates]),dim=-1)
                    # predictions = torch.norm(p_log_map(model.Eh.weight[query]),2,dim=-1)*torch.norm(p_log_map(model.Eh.weight[candidates]),2,dim=-1)*torch.cosine_similarity(p_log_map(model.Eh.weight[query]), p_log_map(model.Eh.weight[candidates]))
                    # predictions = -1*calculate_dist(model.Eh.weight[query], model.Eh.weight[candidates])+model.bias[0][query]+model.bias[1][candidates]

                    for m in range(predictions.shape[0]):
                    #     data_point_temp = copy.deepcopy(list(test_data_batch[m]))
                    #     raw_fact = [data_point_temp[1]] + [data_point_temp[i] for i in
                    #                                        range(0, len(data_point_temp), 2)]
                    #     raw_fact[int(i / 2) + 1] = (int(i / 2) + 1) * 111111

                    #     if i == 2 or i == 0:
                    #         filt = er_vocab[(test_data_batch[m][0], test_data_batch[m][1])]
                    #         # filt = list(set(sr_vocab[tuple(raw_fact)]))
                    #         filt = []
                    #     else:
                    #         filt = er_vocab[tuple(raw_fact)]
                    #         filt = []

                        # target_value = predictions[m, e_i[m]].item()
                    
                        prediction_m = predictions[m, :]

                        # prediction_m[filt] = -np.Inf

                        # remove self_loop
                        # prediction_m[test_data_batch[m][0]] = -np.Inf

                        # prediction_m[e_i[m]] = target_value

                        sort_values, sort_idxs = torch.sort(prediction_m, descending=True)

                        sort_idxs = sort_idxs.cpu().numpy()
                        if i == 2 and j >= (len(test_data_idxs) / 2):
                            topten = [ q2c[test_data_batch[0][2]][kk] for kk in sort_idxs[:100]]
                        else:
                            topten = sort_idxs[:100]

                        fact_string = []
                        for k in range(0,len(test_data_batch[m]),2):
                            fact_string.append(self.idxs2entity[test_data_batch[m][k]])
                            fact_string.append(self.idxs2relation[test_data_batch[m][k+1]])

                        if i == 2 and j >= (len(test_data_idxs) / 2):
                            
                            top = [self.idxs2entity[item] for item in topten]
                        else:
                            top = [self.idxs2entity[item] for item in topten.tolist()]

                        #if top[0]==self.idxs2entity[e_i[m].item()] :
                            #logger.info("position"+str(i)+" "+str(m+j)+" golden truth:"+self.idxs2entity[e_i[m].item()]+" fact: "+" ".join(fact_string)+" top ten: "+" ".join(top))
                        # print(fact_string)
                        # print(top)

                        # rank = np.where(sort_idxs == e_i[m].item())[0][0]
                        
                        
                        if i==2 and j >= (len(test_data_idxs) / 2):
                            golden_truth = self.idxs2entity[e_i[m].item()].replace("queryid", "")
                            new_top = [ int(candi.split("-")[-1]) for candi in top]
                            # logger.info("position"+str(i)+" "+str(m+j)+" golden truth:"+self.idxs2entity[e_i[m].item()]+" fact: "+" ".join(fact_string)+" top ten: "+" ".join(top))
                            output_dict[golden_truth]=new_top

        # with open("./data/LeCard-fine-grained/"+"knn600.json","w") as f:
        #     json.dump(output_dict, f)
        
        os.makedirs('./result/NARY/', exist_ok=True)
        saved_file = './result/NARY/event_vector_result.json'
        with open(saved_file, 'w') as f:
            json.dump(output_dict, f, ensure_ascii=False)

        met = Metric("./input_data")

        # logger.info("========================================")
        # logger.info('BM25')
        # logger.info(met.pred_single_path('./result/Unsupervised/bm25_top100.json'))  # path to the predicted files
        
        # logger.info("========================================")
        # logger.info('tf-idf')
        # logger.info(met.pred_single_path('./result/Unsupervised/tfidf_top100.json'))
        
        # logger.info("========================================")
        # # BERT with event
        # logger.info('lmr')
        # logger.info(met.pred_single_path('./result/Unsupervised/lm_top100.json'))
        
        # logger.info("========================================")
        # logger.info("Bag_of_Event")
        # logger.info(met.pred_single_path('./result/event/event_vector_result.json'))
        
        # logger.info("========================================")
        # logger.info('PolygonE  Prohib 765 epoch')
        # logger.info(met.pred_single_path('./result/NARY/Centroid400.json'))
        
        # logger.info("=====================================================")
        logger.info('PolygonE This model')
        output_dict = met.pred_single_path(saved_file)
        logger.info(output_dict)
        logger.info("=====================================================")

    
    # no-reverse
    def evaluate_batch_ckpt(self, model, data):

        hits = []
        ranks = []

        hits_nry = []
        ranks_nry = []

        head_and_tail_hits = []
        head_and_tail_rank = []

        head_and_tail_hits_2ry = []
        head_and_tail_rank_2ry = []

        head_and_tail_hits_nry = []
        head_and_tail_rank_nry = []

        value_hits_nry = []
        value_rank_nry = []

        for n in range(10):
            hits.append([])
            hits_nry.append([])
            head_and_tail_hits.append([])
            head_and_tail_hits_2ry.append([])
            head_and_tail_hits_nry.append([])
            value_hits_nry.append([])

        test_data_group_by_arity = data.load_data_group_by_arity(data.test_data)
        logger.info("Starting testing...")

        all_fact = self.get_data_idxs(data.data)
        # er_vocab, ht_vocab = self.get_er_vocab(all_fact)
        er_vocab, ht_vocab = [],[]
        q2c = self.get_to_test(data.toTest)
        output_dict = {}

        for arity_data in test_data_group_by_arity[:]:

            hits_this_arity = []
            ranks_this_arity = []
            for p in range(10):
                hits_this_arity.append([])

            test_data_idxs = self.get_data_idxs(arity_data)
            fact_length = len(test_data_idxs[0])

            for j in range(0, len(test_data_idxs), self.test_batch_size):
                # print(j)
                # for i in (list(range(2, fact_length, 2))):
                for i in (list(range(0, 2, 2))):
                    if i > 0:
                        continue

                    if j + self.test_batch_size < len(test_data_idxs):
                        test_data_batch = np.array(test_data_idxs[j:j + self.test_batch_size])
                    else:
                        test_data_batch = np.array(test_data_idxs[j:len(test_data_idxs)])
                    # test_data_batch = [np.array(test_data_idxs[j:j + self.test_batch_size])]
                    temp = []

                    # if i==2 and j >=(len(test_data_idxs) / 2):
                    if i==0:
                        # all_samples = np.repeat(range(len(data.entities)), test_data_batch.shape[0]).reshape(-1,
                        #                                                                               test_data_batch.shape[0]).T
                        all_samples = np.repeat(q2c[test_data_batch[0][0]], test_data_batch.shape[0]).reshape(-1,test_data_batch.shape[0]).T
                    else: 
                        all_samples = np.repeat(range(len(data.entities)), test_data_batch.shape[0]).reshape(-1,
                                                                                                      test_data_batch.shape[0]).T
                    
                    # if i==2 and j >=(len(test_data_idxs) / 2):
                    if i==0:
                        for k in range(len(test_data_idxs[0])):
                            temp.append(
                                torch.LongTensor(np.tile(np.array([test_data_batch[:, k]]).T, (1, len(q2c[test_data_batch[0][0]])))))
                    else:
                        for k in range(len(test_data_idxs[0])):
                            temp.append(
                                torch.LongTensor(np.tile(np.array([test_data_batch[:, k]]).T, (1, len(data.entities)))))
                    
                    temp[i] = torch.LongTensor(all_samples)

                    e_i = test_data_batch[:, i]
                    temp.append(i)
                    temp.append(True)

                    predictions = model.forward(temp)

                    for m in range(predictions.shape[0]):
                        # data_point_temp = copy.deepcopy(list(test_data_batch[m]))
                        # raw_fact = [data_point_temp[1]] + [data_point_temp[i] for i in
                        #                                    range(0, len(data_point_temp), 2)]
                        # raw_fact[int(i / 2) + 1] = (int(i / 2) + 1) * 111111

                        if i == 2 or i == 0:
                            # filt = er_vocab[(test_data_batch[m][0], test_data_batch[m][1])]
                            # filt = list(set(sr_vocab[tuple(raw_fact)]))
                            filt = []
                        else:
                            # filt = er_vocab[tuple(raw_fact)]
                            filt = []

                        # target_value = predictions[m, e_i[m]].item()

                        prediction_m = predictions[m, :]

                        # prediction_m[filt] = -np.Inf

                        # remove self_loop
                        # prediction_m[test_data_batch[m][0]] = -np.Inf

                        # prediction_m[e_i[m]] = target_value

                        sort_values, sort_idxs = torch.sort(prediction_m, descending=True)

                        sort_idxs = sort_idxs.cpu().numpy()
                        # if i==2 and j >=(len(test_data_idxs) / 2):
                        if i==0:
                            topten = [ q2c[test_data_batch[0][0]][kk] for kk in sort_idxs]
                        else:
                            topten = sort_idxs

                        fact_string = []
                        for k in range(0,len(test_data_batch[m]),2):
                            fact_string.append(self.idxs2entity[test_data_batch[m][k]])
                            fact_string.append(self.idxs2relation[test_data_batch[m][k+1]])

                        # if i==2 and j >=(len(test_data_idxs) / 2):
                        if i==0:
                            
                            top = [self.idxs2entity[item] for item in topten]
                        else:
                            top = [self.idxs2entity[item] for item in topten.tolist()]

                        #if top[0]==self.idxs2entity[e_i[m].item()] :
                            #logger.info("position"+str(i)+" "+str(m+j)+" golden truth:"+self.idxs2entity[e_i[m].item()]+" fact: "+" ".join(fact_string)+" top ten: "+" ".join(top))
                        # print(fact_string)
                        # print(top)

                        # rank = np.where(sort_idxs == e_i[m].item())[0][0]
                        rank = 1
                        
                        # if i==2 and j >=(len(test_data_idxs) / 2):
                        if i==0 :
                            golden_truth = self.idxs2entity[e_i[m].item()].replace("queryid", "")
                            new_top = [ int(candi.split("-")[-1]) for candi in top]
                            # logger.info("position"+str(i)+" "+str(m+j)+" golden truth:"+self.idxs2entity[e_i[m].item()]+" fact: "+" ".join(fact_string)+" top ten: "+" ".join(top))
                            output_dict[golden_truth]=new_top

                        # 计算以后，管他三七二十一，整体的rank先加一
                        # ranks_this_arity.append(rank + 1)

                        # for hits_level in range(10):

                        #     # 整体的，不管头尾还是属性，都要计算一下
                        #     if rank <= hits_level:
                        #         hits_this_arity[hits_level].append(1.0)
                        #     else:
                        #         hits_this_arity[hits_level].append(0.0)

                        # if i != 1:
                        #     ranks.append(rank + 1)
                        #     if fact_length == 4:
                        #         head_and_tail_rank.append(rank + 1)
                        #         head_and_tail_rank_2ry.append(rank + 1)

                        #     if fact_length > 4:
                        #         ranks_nry.append(rank + 1)
                        #         if i == 2 or i == 0:
                        #             head_and_tail_rank.append(rank + 1)
                        #             head_and_tail_rank_nry.append(rank + 1)
                        #         else:
                        #             value_rank_nry.append(rank + 1)

                        #     for hits_level in range(10):

                        #         # 整体的，不管头尾还是属性，都要计算一下
                        #         if rank <= hits_level:
                        #             hits[hits_level].append(1.0)
                        #         else:
                        #             hits[hits_level].append(0.0)

                        #         # 进行细致分类
                        #         if fact_length == 4:
                        #             # 如果是3元以上fact,则应该对head_and_tail_hits以及head_and_tail_hits_2ry这两张表进行操作
                        #             if rank <= hits_level:
                        #                 head_and_tail_hits[hits_level].append(1.0)
                        #                 head_and_tail_hits_2ry[hits_level].append(1.0)
                        #             else:
                        #                 head_and_tail_hits[hits_level].append(0.0)
                        #                 head_and_tail_hits_2ry[hits_level].append(0.0)

                        #         if fact_length > 4:

                        #             # 先计算总的
                        #             if rank <= hits_level:
                        #                 hits_nry[hits_level].append(1.0)
                        #             else:
                        #                 hits_nry[hits_level].append(0.0)

                        #             # 如果是多元fact
                        #             if i == 2 or i == 0:
                        #                 # 且测的是头尾实体，则在表head_and_tail_hits和head_and_tail_hits_nry这两张表上进行操作
                        #                 if rank <= hits_level:
                        #                     head_and_tail_hits[hits_level].append(1.0)
                        #                     head_and_tail_hits_nry[hits_level].append(1.0)
                        #                 else:
                        #                     head_and_tail_hits[hits_level].append(0.0)
                        #                     head_and_tail_hits_nry[hits_level].append(0.0)

                        #             if i > 2:
                        #                 if rank <= hits_level:
                        #                     # 如果是多元fact且测属性，就在属性表上进行操作
                        #                     value_hits_nry[hits_level].append(1.0)
                        #                 else:
                        #                     value_hits_nry[hits_level].append(0.0)

            # logger.info('{0} arity data: [MRR Hits@10 Hits@5 Hits@3 Hits@1]: {1} {2} {3} {4} {5}'.format(
            #     str(int((fact_length + 1) / 2)),
            #     str(np.mean(1. / np.array(ranks_this_arity))),
            #     str(np.mean(hits_this_arity[9])),
            #     str(np.mean(hits_this_arity[4])),
            #     str(np.mean(hits_this_arity[2])),
            #     str(np.mean(hits_this_arity[0])))

            # )
     
        os.makedirs('./result/NARY/', exist_ok=True)
        saved_file = './result/NARY/event_vector_result.json'
        with open(saved_file, 'w') as f:
            json.dump(output_dict, f, ensure_ascii=False)

        met = Metric("./input_data")

        logger.info("========================================")
        logger.info('BM25')
        logger.info(met.pred_single_path('./result/Unsupervised/bm25_top100.json'))  # path to the predicted files
        
        logger.info("========================================")
        logger.info('tf-idf')
        logger.info(met.pred_single_path('./result/Unsupervised/tfidf_top100.json'))
        
        logger.info("========================================")
        # BERT with event
        logger.info('lmr')
        logger.info(met.pred_single_path('./result/Unsupervised/lm_top100.json'))
        
        logger.info("========================================")
        logger.info("Bag_of_Event")
        logger.info(met.pred_single_path('./result/event/event_vector_result.json'))
        
        logger.info("========================================")
        logger.info('PolygonE  Prohib 765 epoch')
        logger.info(met.pred_single_path('./result/NARY/Centroid400.json'))
        
        logger.info("=====================================================")
        logger.info('PolygonE This model')
        results = met.pred_single_path(saved_file)
        logger.info(results)
        logger.info("=====================================================")
            
        # logger.info("entity:")
        # logger.info('Overall resuts: [MRR Hits@10 Hits@5 Hits@3 Hits@1]:{0} {1} {2} {3} {4}'.format(
        #     str(np.mean(1. / np.array(ranks))), str(np.mean(hits[9])), str(np.mean(hits[4])), str(np.mean(hits[2])),
        #     str(np.mean(hits[0]))))
        # logger.info('head_tail_all: [MRR Hits@10 Hits@5 Hits@3 Hits@1]:{0} {1} {2} {3} {4}'.format(
        #     str(np.mean(1. / np.array(head_and_tail_rank))), str(np.mean(head_and_tail_hits[9])),
        #     str(np.mean(head_and_tail_hits[4])), str(np.mean(head_and_tail_hits[2])),
        #     str(np.mean(head_and_tail_hits[0]))))
        # logger.info('head_and_tail_2ry: [MRR Hits@10 Hits@5 Hits@3 Hits@1]:{0} {1} {2} {3} {4}'.format(
        #     str(np.mean(1. / np.array(head_and_tail_rank_2ry))), str(np.mean(head_and_tail_hits_2ry[9])),
        #     str(np.mean(head_and_tail_hits_2ry[4])), str(np.mean(head_and_tail_hits_2ry[2])),
        #     str(np.mean(head_and_tail_hits_2ry[0]))))
        # logger.info('head_and_tail_nry: [MRR Hits@10 Hits@5 Hits@3 Hits@1]:{0} {1} {2} {3} {4}'.format(
        #     str(np.mean(1. / np.array(head_and_tail_rank_nry))), str(np.mean(head_and_tail_hits_nry[9])),
        #     str(np.mean(head_and_tail_hits_nry[4])), str(np.mean(head_and_tail_hits_nry[2])),
        #     str(np.mean(head_and_tail_hits_nry[0]))))
        # logger.info('affiliated_entity: [MRR Hits@10 Hits@5 Hits@3 Hits@1]:{0} {1} {2} {3} {4}'.format(
        #     str(np.mean(1. / np.array(value_rank_nry))), str(np.mean(value_hits_nry[9])),
        #     str(np.mean(value_hits_nry[4])), str(np.mean(value_hits_nry[2])), str(np.mean(value_hits_nry[0]))))
        # logger.info('n_ary_fact_overall: [MRR Hits@10 Hits@5 Hits@3 Hits@1]:{0} {1} {2} {3} {4}'.format(
        #     str(np.mean(1. / np.array(ranks_nry))), str(np.mean(hits_nry[9])), str(np.mean(hits_nry[4])),
        #     str(np.mean(hits_nry[2])), str(np.mean(hits_nry[0]))))


    def evaluate_batch(self, model, data):

        hits = []
        ranks = []

        hits_nry = []
        ranks_nry = []

        head_and_tail_hits = []
        head_and_tail_rank = []

        head_and_tail_hits_2ry = []
        head_and_tail_rank_2ry = []

        head_and_tail_hits_nry = []
        head_and_tail_rank_nry = []

        value_hits_nry = []
        value_rank_nry = []

        for n in range(10):
            hits.append([])
            hits_nry.append([])
            head_and_tail_hits.append([])
            head_and_tail_hits_2ry.append([])
            head_and_tail_hits_nry.append([])
            value_hits_nry.append([])



        test_data_group_by_arity = d.load_data_group_by_arity(data)
        logger.info("Starting testing...")

        all_fact = self.get_data_idxs(d.data)
        er_vocab, ht_vocab = self.get_er_vocab(all_fact)


        for arity_data in test_data_group_by_arity:

            hits_this_arity = []
            ranks_this_arity = []
            for p in range(10):
                hits_this_arity.append([])


            test_data_idxs = self.get_data_idxs(arity_data)
            fact_length = len(test_data_idxs[0])

            for j in range(0, len(test_data_idxs), self.test_batch_size):

                for i in (list(range(2, fact_length, 2))):
                    if i > 2 and j < (len(test_data_idxs) / 2):
                        continue


                    if j + self.test_batch_size < len(test_data_idxs):
                        test_data_batch = np.array(test_data_idxs[j:j + self.test_batch_size])
                    else:
                        test_data_batch = np.array(test_data_idxs[j:len(test_data_idxs)])

                    temp = []

                    all_samples = np.repeat(range(len(d.entities)), test_data_batch.shape[0]).reshape(-1,test_data_batch.shape[0]).T
                    for k in range(len(test_data_idxs[0])):
                        temp.append(
                            torch.LongTensor(np.tile(np.array([test_data_batch[:, k]]).T, (1, len(d.entities)))))
                    temp[i] = torch.LongTensor(all_samples)

                    e_i = test_data_batch[:, i]
                    temp.append(i)
                    temp.append(True)

                    predictions = model.forward(temp)

                    for m in range(predictions.shape[0]):
                        data_point_temp = copy.deepcopy(list(test_data_batch[m]))
                        raw_fact = [data_point_temp[1]]+[data_point_temp[i] for i in range(0,len(data_point_temp),2)]
                        raw_fact[int(i/2)+1]=(int(i/2)+1)*111111

                        if i == 2 or i==0:
                            filt = er_vocab[(test_data_batch[m][0], test_data_batch[m][1])]
                            # filt = er_vocab[tuple(raw_fact)]
                        else:
                            filt = er_vocab[tuple(raw_fact)]

                        target_value = predictions[m,e_i[m]].item()

                        prediction_m = predictions[m,:]

                        prediction_m[filt] = -np.Inf

                        # remove self_loop
                        prediction_m[test_data_batch[m][0]] = -np.Inf

                        prediction_m[e_i[m]] = target_value

                        sort_values, sort_idxs = torch.sort(prediction_m, descending=True)

                        sort_idxs = sort_idxs.cpu().numpy()
                        rank = np.where(sort_idxs == e_i[m].item())[0][0]

                        # 计算以后，管他三七二十一，整体的rank先加一
                        ranks_this_arity.append(rank+1)

                        for hits_level in range(10):

                            # 整体的，不管头尾还是属性，都要计算一下
                            if rank <= hits_level:
                                hits_this_arity[hits_level].append(1.0)
                            else:
                                hits_this_arity[hits_level].append(0.0)

                        if i != 1:
                            ranks.append(rank + 1)
                            if fact_length == 4:
                                head_and_tail_rank.append(rank + 1)
                                head_and_tail_rank_2ry.append(rank + 1)

                            if fact_length > 4:
                                ranks_nry.append(rank + 1)
                                if i == 2 or i==0:
                                    head_and_tail_rank.append(rank + 1)
                                    head_and_tail_rank_nry.append(rank + 1)
                                else:
                                    value_rank_nry.append(rank + 1)

                            for hits_level in range(10):

                                # 整体的，不管头尾还是属性，都要计算一下
                                if rank <= hits_level:
                                    hits[hits_level].append(1.0)
                                else:
                                    hits[hits_level].append(0.0)

                                # 进行细致分类
                                if fact_length == 4:
                                    # 如果是3元以上fact,则应该对head_and_tail_hits以及head_and_tail_hits_2ry这两张表进行操作
                                    if rank <= hits_level:
                                        head_and_tail_hits[hits_level].append(1.0)
                                        head_and_tail_hits_2ry[hits_level].append(1.0)
                                    else:
                                        head_and_tail_hits[hits_level].append(0.0)
                                        head_and_tail_hits_2ry[hits_level].append(0.0)

                                if fact_length > 4:

                                    # 先计算总的
                                    if rank <= hits_level:
                                        hits_nry[hits_level].append(1.0)
                                    else:
                                        hits_nry[hits_level].append(0.0)

                                    # 如果是多元fact
                                    if i == 2 or i==0:
                                        # 且测的是头尾实体，则在表head_and_tail_hits和head_and_tail_hits_nry这两张表上进行操作
                                        if rank <= hits_level:
                                            head_and_tail_hits[hits_level].append(1.0)
                                            head_and_tail_hits_nry[hits_level].append(1.0)
                                        else:
                                            head_and_tail_hits[hits_level].append(0.0)
                                            head_and_tail_hits_nry[hits_level].append(0.0)

                                    if i > 2:
                                        if rank <= hits_level:
                                            # 如果是多元fact且测属性，就在属性表上进行操作
                                            value_hits_nry[hits_level].append(1.0)
                                        else:
                                            value_hits_nry[hits_level].append(0.0)

            logger.info('{0} arity data: [MRR Hits@10 Hits@5 Hits@3 Hits@1]: {1} {2} {3} {4} {5}'.format(
                str(int((fact_length+1)/2)),
                str(np.mean(1. / np.array(ranks_this_arity))),
                str(np.mean(hits_this_arity[9])),
                str(np.mean(hits_this_arity[4])),
                str(np.mean(hits_this_arity[2])),
                str(np.mean(hits_this_arity[0])))

            )

        logger.info("entity:")
        logger.info('Overall resuts: [MRR Hits@10 Hits@5 Hits@3 Hits@1]:{0} {1} {2} {3} {4}'.format(
            str(np.mean(1. / np.array(ranks))), str(np.mean(hits[9])), str(np.mean(hits[4])), str(np.mean(hits[2])),
            str(np.mean(hits[0]))))
        logger.info('head_tail_all: [MRR Hits@10 Hits@5 Hits@3 Hits@1]:{0} {1} {2} {3} {4}'.format(
            str(np.mean(1. / np.array(head_and_tail_rank))), str(np.mean(head_and_tail_hits[9])),
            str(np.mean(head_and_tail_hits[4])), str(np.mean(head_and_tail_hits[2])),
            str(np.mean(head_and_tail_hits[0]))))
        logger.info('head_and_tail_2ry: [MRR Hits@10 Hits@5 Hits@3 Hits@1]:{0} {1} {2} {3} {4}'.format(
            str(np.mean(1. / np.array(head_and_tail_rank_2ry))), str(np.mean(head_and_tail_hits_2ry[9])),
            str(np.mean(head_and_tail_hits_2ry[4])), str(np.mean(head_and_tail_hits_2ry[2])),
            str(np.mean(head_and_tail_hits_2ry[0]))))
        logger.info('head_and_tail_nry: [MRR Hits@10 Hits@5 Hits@3 Hits@1]:{0} {1} {2} {3} {4}'.format(
            str(np.mean(1. / np.array(head_and_tail_rank_nry))), str(np.mean(head_and_tail_hits_nry[9])),
            str(np.mean(head_and_tail_hits_nry[4])), str(np.mean(head_and_tail_hits_nry[2])),
            str(np.mean(head_and_tail_hits_nry[0]))))
        logger.info('affiliated_entity: [MRR Hits@10 Hits@5 Hits@3 Hits@1]:{0} {1} {2} {3} {4}'.format(
            str(np.mean(1. / np.array(value_rank_nry))), str(np.mean(value_hits_nry[9])),
            str(np.mean(value_hits_nry[4])), str(np.mean(value_hits_nry[2])), str(np.mean(value_hits_nry[0]))))
        logger.info('n_ary_fact_overall: [MRR Hits@10 Hits@5 Hits@3 Hits@1]:{0} {1} {2} {3} {4}'.format(
            str(np.mean(1. / np.array(ranks_nry))), str(np.mean(hits_nry[9])), str(np.mean(hits_nry[4])),
            str(np.mean(hits_nry[2])), str(np.mean(hits_nry[0]))))



    def evaluate_for_checkpoint(self, model, d):
        hits = []
        ranks = []

        hits_nry = []
        ranks_nry = []

        head_and_tail_hits = []
        head_and_tail_rank = []

        head_and_tail_hits_2ry = []
        head_and_tail_rank_2ry = []

        head_and_tail_hits_nry = []
        head_and_tail_rank_nry = []

        value_hits_nry = []
        value_rank_nry = []

        relation_hits = []
        relation_ranks = []

        relation_hits_2ry = []
        relation_ranks_2ry = []

        relation_hits_nry = []
        relation_ranks_nry = []

        for n in range(10):
            hits.append([])
            hits_nry.append([])
            head_and_tail_hits.append([])
            head_and_tail_hits_2ry.append([])
            head_and_tail_hits_nry.append([])
            value_hits_nry.append([])

            relation_hits.append([])
            relation_hits_2ry.append([])
            relation_hits_nry.append([])

        test_data_idxs = self.get_data_idxs(d.test_data)

        logger.info("Number of data points: %d" % len(test_data_idxs))

        count = 0

        


        # 一条一条测
        all_fact = self.get_data_idxs(d.data)
        sr_vocab, ht_vocab = self.get_er_vocab(all_fact)
        
        result = []
        for j in range(0, len(test_data_idxs)//2):
            fact_length = len(test_data_idxs[j])
            print(j)
            # for i in (list(range(2, fact_length, 2))):
            for i in (list(range(4, 6, 2))):
                # if i > 2 and j < (len(test_data_idxs) / 2):
                if i>4:
                    continue


                temp = []
                data_point = test_data_idxs[j]
                for k in range(len(test_data_idxs[j])):
                    temp.append(torch.tensor(data_point[k]))

                if self.cuda:
                    for p in range(len(test_data_idxs[j])):
                        temp[p] = temp[p].cuda()

                e_i = temp[i]
                if i != 1:
                    temp = [item.repeat(len(d.entities)) for item in temp]
                    temp[i] = torch.tensor(range(len(d.entities))).cuda()
                else:
                    temp = [item.repeat(len(d.relations)) for item in temp]
                    temp[i] = torch.tensor(range(len(d.relations))).cuda()

                temp.append(i)
                temp.append(True)
                predictions_s = model.forward(temp)

                data_point_temp = copy.deepcopy(list(data_point))
                del data_point_temp[i]

                if i == 2 or i == 0:
                    filt = list(set(sr_vocab[(data_point[0], data_point[1])]))
                    # filt = list(set(sr_vocab[tuple(data_point_temp)]))
                elif i == 1:
                    filt = list(set(ht_vocab[(data_point[0], data_point[2])]))
                else:
                    filt = list(set(sr_vocab[tuple(data_point_temp)]))

                target_value = predictions_s[e_i].item()

                if i != 1:
                    #  remove self-loop
                    predictions_s[data_point[0]] = -np.Inf

                    predictions_s[filt] = -np.Inf

                    predictions_s[e_i] = target_value


                sort_values, sort_idxs = torch.sort(predictions_s, descending=True)

                top_ten = sort_idxs[:10].cpu().numpy().tolist()
                head = test_data_idxs[j][0]
                rel  = test_data_idxs[j][5]
                top_ten.insert(0,rel)
                top_ten.insert(0,head)

                
                result.append(top_ten)


                sort_idxs = sort_idxs.cpu().numpy()
                rank = np.where(sort_idxs == e_i.item())[0][0]

                # 计算以后，管他三七二十一，整体的rank先加一

                if i != 1:
                    ranks.append(rank + 1)
                    if fact_length == 3:
                        head_and_tail_rank.append(rank + 1)
                        head_and_tail_rank_2ry.append(rank + 1)

                    if fact_length > 3:
                        ranks_nry.append(rank + 1)
                        if i == 2:
                            head_and_tail_rank.append(rank + 1)
                            head_and_tail_rank_nry.append(rank + 1)
                        else:
                            value_rank_nry.append(rank + 1)

                    for hits_level in range(10):

                        # 整体的，不管头尾还是属性，都要计算一下
                        if rank <= hits_level:
                            hits[hits_level].append(1.0)
                        else:
                            hits[hits_level].append(0.0)

                        # 进行细致分类
                        if fact_length == 3:
                            # 如果是3元以上fact,则应该对head_and_tail_hits以及head_and_tail_hits_2ry这两张表进行操作
                            if rank <= hits_level:
                                head_and_tail_hits[hits_level].append(1.0)
                                head_and_tail_hits_2ry[hits_level].append(1.0)
                            else:
                                head_and_tail_hits[hits_level].append(0.0)
                                head_and_tail_hits_2ry[hits_level].append(0.0)

                        if fact_length > 3:

                            # 先计算总的
                            if rank <= hits_level:
                                hits_nry[hits_level].append(1.0)
                            else:
                                hits_nry[hits_level].append(0.0)

                            # 如果是多元fact
                            if i == 2:
                                # 且测的是头尾实体，则在表head_and_tail_hits和head_and_tail_hits_nry这两张表上进行操作
                                if rank <= hits_level:
                                    head_and_tail_hits[hits_level].append(1.0)
                                    head_and_tail_hits_nry[hits_level].append(1.0)
                                else:
                                    head_and_tail_hits[hits_level].append(0.0)
                                    head_and_tail_hits_nry[hits_level].append(0.0)

                            if i > 2:
                                if rank <= hits_level:
                                    # 如果是多元fact且测属性，就在属性表上进行操作
                                    value_hits_nry[hits_level].append(1.0)
                                else:
                                    value_hits_nry[hits_level].append(0.0)

        with open("result_nary.tsv","w") as f_p:
                for a_top_ten in result:
                    rel = self.idxs2relation[a_top_ten[1]]
                    top_ent = [self.idxs2entity[a_top_ten[aaa]] for aaa in range(len(a_top_ten))]
                    top_ent[1]= rel
                    top_string = '\t'.join(top_ent)
                    f_p.write(top_string+"\n")
        
        logger.info("entity:")
        logger.info('GTED compare [MRR Hits@10 Hits@5 Hits@3 Hits@1]:{0} {1} {2} {3} {4}'.format(
            str(np.mean(1. / np.array(ranks))), str(np.mean(hits[9])), str(np.mean(hits[4])), str(np.mean(hits[2])),
            str(np.mean(hits[0]))))
        logger.info('head_tail_all [MRR Hits@10 Hits@5 Hits@3 Hits@1]:{0} {1} {2} {3} {4}'.format(
            str(np.mean(1. / np.array(head_and_tail_rank))), str(np.mean(head_and_tail_hits[9])),
            str(np.mean(head_and_tail_hits[4])), str(np.mean(head_and_tail_hits[2])),
            str(np.mean(head_and_tail_hits[0]))))
        logger.info('head_and_tail_2ry [MRR Hits@10 Hits@5 Hits@3 Hits@1]:{0} {1} {2} {3} {4}'.format(
            str(np.mean(1. / np.array(head_and_tail_rank_2ry))), str(np.mean(head_and_tail_hits_2ry[9])),
            str(np.mean(head_and_tail_hits_2ry[4])), str(np.mean(head_and_tail_hits_2ry[2])),
            str(np.mean(head_and_tail_hits_2ry[0]))))
        logger.info('head_and_tail_nry [MRR Hits@10 Hits@5 Hits@3 Hits@1]:{0} {1} {2} {3} {4}'.format(
            str(np.mean(1. / np.array(head_and_tail_rank_nry))), str(np.mean(head_and_tail_hits_nry[9])),
            str(np.mean(head_and_tail_hits_nry[4])), str(np.mean(head_and_tail_hits_nry[2])),
            str(np.mean(head_and_tail_hits_nry[0]))))
        logger.info('affiliated_entity [MRR Hits@10 Hits@5 Hits@3 Hits@1]:{0} {1} {2} {3} {4}'.format(
            str(np.mean(1. / np.array(value_rank_nry))), str(np.mean(value_hits_nry[9])),
            str(np.mean(value_hits_nry[4])), str(np.mean(value_hits_nry[2])), str(np.mean(value_hits_nry[0]))))
        logger.info('n_ary_fact_overall [MRR Hits@10 Hits@5 Hits@3 Hits@1]:{0} {1} {2} {3} {4}'.format(
            str(np.mean(1. / np.array(ranks_nry))), str(np.mean(hits_nry[9])), str(np.mean(hits_nry[4])),
            str(np.mean(hits_nry[2])), str(np.mean(hits_nry[0]))))

    def get_scr_golden(self, device):
        golden = json.load(open("./input_data/label/label_top30_dict.json", "r"))
        candidates = json.load(open("./input_data/label/candidate_dict.json", "r"))
        lst_similar, lst_unsimilar = [], []
        [lst_similar.extend(
            ["queryid" + str(q) + "#" + "candidate-for-queryid" + str(q) + "-" + str(idx) for idx in d]
        ) for q, d in golden.items()]
        [lst_unsimilar.extend(
            ["queryid" + str(q) + "#" + "candidate-for-queryid" + str(q) + "-" + str(idx) for idx in d]
        ) for q, d in candidates.items()]
        cons = [
            [self.entity2idxs.get(pair.split("#")[0]), self.entity2idxs.get(pair.split("#")[1]), 0]
            for pair in list(set(lst_unsimilar) - set(lst_similar))
            if (pair.split("#")[0] in self.entity2idxs) and (pair.split("#")[1] in self.entity2idxs)
        ]
        pros = [
            [self.entity2idxs.get(pair.split("#")[0]), self.entity2idxs.get(pair.split("#")[1]), 1]
            for pair in list(set(lst_similar))
            if (pair.split("#")[0] in self.entity2idxs) and (pair.split("#")[1] in self.entity2idxs)
        ]
        golden_data = torch.tensor(pros + cons)
        return golden_data[:, :2].long().to(device), golden_data[:, 2].double().to(device)

    def train_and_eval(self):
        # d是整体的数据集
        logger.info("Training the %s model..." % self.model)
        self.entity2idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.idxs2entity = {i: d.entities[i] for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}
        self.idxs2relation = {i:d.entities[i] for i in range(len(d.relations))}

        with open("idxs2entity.json","w") as fent, open("idxs2relation.json","w") as frel:
            json.dump(self.entity2idxs,fent)
            json.dump(self.relation_idxs,frel)

        # self.primary_entities_indexes = [self.entity2idxs[item] for item in d.primary_entities]
        # self.affiliated_entities_indexes = [self.entity2idxs[item] for item in d.affilated_entities]

        """
        这一行d.train_data得改成相应维度的数据才可以,这里开始得使用for循环了哦
        """
        bce_loss_fn = torch.nn.BCEWithLogitsLoss()
        contrasiveloss = ContrasiveLoss()
        if self.model == "poincare":
            model = MuRP(d, self.dim, self.device)
            param_names = [name for name, param in model.named_parameters()]
            opt = RiemannianSGD(model.parameters(), lr=self.learning_rate, param_names=param_names)
        elif self.model == "Centroid":
            model = Centroid(d, self.dim, self.device)
            param_names = [name for name, param in model.named_parameters()]
            opt = RiemannianSGD(model.parameters(), lr=self.learning_rate, param_names=param_names)
        elif self.model == "WPolygonE":
            model = WPolygonE(d, self.dim, self.device)
            param_names = [name for name, param in model.named_parameters()]
            opt = RiemannianSGD(model.parameters(), lr=self.learning_rate, param_names=param_names)
        elif self.model == "Bias_Centroid":
            model = Bias_Centroid(d, self.dim, self.device)
            param_names = [name for name, param in model.named_parameters()]
            opt = RiemannianSGD(model.parameters(), lr=self.learning_rate, param_names=param_names)
        elif self.model == "all_Centroid":
            model = all_Centroid(d, self.dim, self.device)
            param_names = [name for name, param in model.named_parameters()]
            opt = RiemannianSGD(model.parameters(), lr=self.learning_rate, param_names=param_names)
        elif self.model == "WE_Centroid" :
            model = WE_centroid(d,self.dim, self.device)
            param_names = [name for name, param in model.named_parameters()]
            opt = optim.Adam(model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
            # opt = optim.Adagrad(model.parameters(),lr=self.learning_rate)
        elif self.model == "LeCentroid":
            model = LeCentroid(d,self.dim, self.device)
            param_names = [name for name, param in model.named_parameters()]
            opt = RiemannianSGD(model.parameters(), lr=self.learning_rate, param_names=param_names)
        elif self.model == "E_Centroid":
            model = E_centroid(d,self.dim, self.device)
            param_names = [name for name, param in model.named_parameters()]
            opt = optim.Adam(model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        elif self.model == "E_dot_Centroid":
            model = E_dot_Centroid(d,self.dim, self.device)
            param_names = [name for name, param in model.named_parameters()]
            opt = optim.Adam(model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        elif self.model == "E_cos_Centroid":
            model = E_cos_Centroid(d,self.dim, self.device)
            param_names = [name for name, param in model.named_parameters()]
            opt = optim.Adam(model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
            # opt = optim.Adagrad(model.parameters(),lr=self.learning_rate)

        # param_names = [name for name, param in model.named_parameters()]

        # opt = RiemannianSGD(model.parameters(), lr=self.learning_rate, param_names=param_names)
        # opt = optim.Adam(model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        # opt = optim.SGD(model.parameters(),lr=self.learning_rate,momentum=0.9)
        # opt = optim.AdamW(model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999),weight_decay=5e-4)

        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info("param:" + name +" "+ str(param.size()))

        if self.cuda:
            model.cuda()

        train_data_group_by_arity = d.load_data_group_by_arity(d.train_data)
        logger.info("Starting training...")
        start_train = time.time()

        scr_pair, scr_target = self.get_scr_golden(self.device) if self.use_scr else (None, None)
        ht_losses, scr_losses, contra_losses = [], [], []

        if 1 == (False * 1):  # 导出embedding用
            model.load_state_dict(torch.load(f'./checkpoint_model_epoch_1000.pth.tar'))
            pickle.dump(p_log_map(model.Eh.weight), open("entity_embedding.pickle", "wb"))
            exit(0)

        for it in range(1, self.num_iterations + 1):
            model.train()
            losses = []
            np.random.shuffle(train_data_group_by_arity)
            for arity_data in train_data_group_by_arity:

                train_data_idxs = self.get_data_idxs(arity_data)

                if it == 1:
                    logger.info("Number of "+str(int((len(train_data_idxs[0])/2)))+"ary facts: "+ str(len(train_data_idxs)))

                np.random.shuffle(train_data_idxs)

                for j in range(0, len(train_data_idxs), self.batch_size):


                    aaaa = np.random.choice([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]+list(range(2, len(train_data_idxs[0]), 2)),size=1)  # 加了reverse,从2开始就可以,随机corrupt一个实体域
                    # i = np.random.choice(list(range(2, len(train_data_idxs[0]), 2)))
                    # i = np.random.choice([2,2]+list(range(2, len(train_data_idxs[0]), 2)))
                    #i = 2
                    for i in aaaa:
                        # second_i = np.random.choice([2]+list(range(2, len(train_data_idxs[0]), 2)))
                        
                        if len(train_data_idxs) > self.batch_size:
                            data_batch = np.array(train_data_idxs[j:j + self.batch_size])
                        else:
                            data_batch = np.array(train_data_idxs[:])

                        # data_batch = np.array(train_data_idxs[j:j + self.batch_size])


                        negsamples = np.random.choice(list(self.entity2idxs.values()),
                                                        size=(data_batch.shape[0], self.nneg))

                        # negsamples_2 = np.random.choice(list(self.entity2idxs.values()),
                        #                                   size=(data_batch.shape[0], self.nneg))
                        # np.tile function: np.tile(a,(2,1)),the first argument  means  把数组竖着扩大两边，1表示横轴不变

                        temp = []

                        for k in range(len(train_data_idxs[0])):
                            temp.append(
                                torch.LongTensor(np.tile(np.array([data_batch[:, k]]).T, (1, negsamples.shape[1] + 1))))
                            # temp[2] = torch.LongTensor(np.concatenate((np.array([data_batch[:, 2]]).T, negsamples_tri), axis=1))
                        temp[i] = torch.LongTensor(np.concatenate((np.array([data_batch[:, i]]).T, negsamples), axis=1))
                        # temp[second_i] = torch.LongTensor(np.concatenate((np.array([data_batch[:, i]]).T, negsamples_2), axis=1))
                    
                        targets = np.zeros(temp[0].shape)
                        # targets[:, 0] = 0.9001  # 所有行的第0个元素取1
                        # targets[:, 1:] = 0.0001
                        targets[:,0]=1
                        targets = torch.DoubleTensor(targets)

                        opt.zero_grad()
                        if self.cuda:
                            targets = targets.cuda()
                            for p in range(len(temp)):
                                temp[p] = temp[p].cuda()

                        # temp.append(i)
                        self.do_test = False
                        temp.append(i)
                        temp.append(self.do_test)

                        # prediction_ht = model.forward(temp)
                        # loss_ht = model.loss(prediction_ht, targets)
                        prediction_ht, scr_score = model.forward(temp, scr_data=scr_pair)
                        loss_ht = bce_loss_fn(prediction_ht, targets)
                        ht_losses.append(loss_ht.item())

                        if self.use_scr:
                            loss_scr = 0.01 * bce_loss_fn(scr_score, scr_target)
                            scr_losses.append(loss_scr.item())
                            loss = loss_ht + loss_scr
                        else:
                            scr_losses.append(0.0)
                            loss = loss_ht

                        loss_contra = contrasiveloss(prediction_ht,temperature=0.8)
                        contra_losses.append(loss_contra.item())
                        # loss_contra = 0
                        loss = loss + 0.0005 * loss_contra

                        loss.backward()
                        opt.step()

            cur_lr = opt.state_dict()['param_groups'][0]['lr']
            now = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
            logger.info("%s epoch:%d ht_loss:%s contra_loss:%s scr_loss:%s lr:%f  training time:%d" % (
                now, it, str(np.mean(ht_losses)), str(np.mean(contra_losses)), str(np.mean(scr_losses)), cur_lr,
                time.time() - start_train
            ))
            model.eval()
            with torch.no_grad():
                if not it % 20 or it== args.num_iterations or it ==601 or it==1202 or it==1404 or it ==1606 or it==400 or it==808 or it==999 or it==1:
                    filepath = os.path.join('./data/' + args.dataset,
                                            'checkpoint_model_epoch_{}.pth.tar'.format(it))  # 最终参数模型
                    torch.save(model.state_dict(), filepath)

                # if not it % 21 and it >=600 or it== args.num_iterations or it==1 or it ==601 or it==1201 or it==1401 or it ==1601 or it==400 or it==800 or it==1000 or it==700:
                if not it%10 or it>=400 or it==1:
                    start_eval = time.time()
                    self.find_KNN(model, d)
                    logger.info("evaluating time:" + str(time.time() - start_eval))
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="LeProhiALL-4-Contra-t05-w0005/data_5fold/query_0", nargs="?",
                        help="Which dataset to use: FB15k-237 or WN18RR.")
    parser.add_argument("--model", type=str, default="LeCentroid", nargs="?",
                        help="Which model to use: poincare or euclidean.")
    # parser.add_argument("--optimizer", type=str, default="rsgd", nargs="?",
    #                     help="rsgd, adagrad or adam")
    parser.add_argument("--num_iterations", type=int, default=1000, nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=64, nargs="?",
                        help="Batch size.")
    parser.add_argument("--nneg", type=int, default=100, nargs="?",
                        help="Number of negative samples.")
    parser.add_argument("--lr", type=float, default=2, nargs="?",
                        help="Learning rate.")
    parser.add_argument("--dim", type=int, default=500, nargs="?",
                        help="Embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                        help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--device_ids", type=str, default="0", nargs="?",
                        help="Which GPU to use.")
    parser.add_argument("--do_train", type=bool, default=True, nargs="?",
                        help="train")
    parser.add_argument("--do_test", type=bool, default=False, nargs="?",
                        help="test.")
    parser.add_argument("--use_scr", type=bool, default=True, nargs="?",
                        help="Include supervised training")
    parser.add_argument("--test_batch_size", type=int, default=1, nargs="?",
                        help="Test Batch Size.")

    args = parser.parse_args()

    args.cuda = args.cuda if torch.cuda.is_available() else False

    for k, v in sorted(vars(args).items()):
        logger.info(str(k) + '=' + str(v))
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    seed = 54
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    d = Data_whole_and_split(data_dir=data_dir)
    # logger.info(str(len(d.affilated_entities)))
    # logger.info(str(len(d.primary_entities)))
    experiment = Experiment(learning_rate=args.lr, batch_size=args.batch_size,
                            num_iterations=args.num_iterations, dim=args.dim,
                            cuda=args.cuda, nneg=args.nneg, model=args.model, device_ids = args.device_ids,
                            do_test=args.do_test, test_batch_size=args.test_batch_size, use_scr=args.use_scr)
    
    experiment.train_and_eval()


