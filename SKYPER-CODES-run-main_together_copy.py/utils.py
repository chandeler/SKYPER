import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch

def artanh(x):
    return 0.5*torch.log((1+x)/(1-x))

def p_exp_map(v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
    return torch.tanh(normv)*v/normv

def p_log_map(v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), 1e-10, 1-1e-5)
    return artanh(normv)*v/normv

def full_p_exp_map(x, v):
    normv = torch.clamp(torch.norm(v, 2, dim=-1, keepdim=True), min=1e-10)
    sqxnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1-1e-5)
    y = torch.tanh(normv/(1-sqxnorm)) * v/normv
    return p_sum(x, y)


def p_sum(x, y):
    sqxnorm = torch.clamp(torch.sum(x * x, dim=-1, keepdim=True), 0, 1-1e-5)
    sqynorm = torch.clamp(torch.sum(y * y, dim=-1, keepdim=True), 0, 1-1e-5)
    dotxy = torch.sum(x*y, dim=-1, keepdim=True)
    numerator = (1+2*dotxy+sqynorm)*x + (1-sqxnorm)*y
    denominator = 1 + 2*dotxy + sqxnorm*sqynorm
    return numerator/denominator
    

def norm_within_one(x):
    return torch.where(torch.norm(x, 2, dim=-1, keepdim=True) >= 1,  x/(torch.norm(x, 2, dim=-1, keepdim=True)**2-1e-5), x)

def p_matric_mul_vec(x,M):
    normx = torch.clamp(torch.norm(x, 2, dim=-1, keepdim=True), min=1e-10)
    Mx = torch.matmul(x,M)
    normMx = torch.clamp(torch.norm(Mx, 2, dim=-1, keepdim=True), min=1e-10)

    result = torch.tanh(normMx/normx*artanh(normx))*Mx/normMx
    return result

def pairinfo_for_single(pair,single,Weight_for_pair):
    pair_1 = p_log_map(pair)
    pair_1 = Weight_for_pair*pair_1
    single_1 = p_log_map(single)
    single_1 = single_1+pair_1
    single_1 = p_exp_map(single_1)
    single_1 = norm_within_one(single_1)
    return single_1

def hyper2tan(pair,single,Weight_for_pair):
    pair_1 = p_log_map(pair)
    pair_1 = Weight_for_pair*pair_1
    single_1 = p_log_map(single)
    single_1 = single_1+pair_1
    # single_1 = p_exp_map(single_1)
    # single_1 = norm_within_one(single_1)
    return single_1

def pairinfo_for_single_with_bias(pair,single,Weight_for_pair,bias):
    pair_1 = p_log_map(pair)
    pair_1 = Weight_for_pair*pair_1
    single_1 = p_log_map(single)

    single_1 = single_1+pair_1+bias
    single_1 = p_exp_map(single_1)
    single_1 = norm_within_one(single_1)
    return single_1

def calculate_dist(u,v):

    sqdist = (2. * artanh(torch.clamp(torch.norm(p_sum(-u, v), 2, dim=-1), 1e-10, 1 - 1e-5)))**2

    return sqdist

def cal_vector_length(gyrovector):
    sqdist = (2. * artanh(torch.clamp(torch.norm(gyrovector, 2, dim=-1), 1e-10, 1 - 1e-5)))

    return sqdist



def hyper_tanget_hyper(u, weight_for_head):
    u_e = p_log_map(u)
    u_e = u_e*weight_for_head
    u_e = p_exp_map(u_e)
    u_e = norm_within_one(u_e)
    return u_e

def cal_midpoint(x,y):

    sqnormx = torch.clamp(torch.sum(x**2, dim=-1, keepdim=True), 0, 1 - 1e-5)
    sqnormy = torch.clamp(torch.sum(y**2, dim=-1, keepdim=True), 0, 1 - 1e-5)
    gamma_x_2 = 1/(1-sqnormx)
    gamma_y_2 = 1/(1-sqnormy)
    mid = (gamma_x_2*x+gamma_y_2*y)/(gamma_x_2+gamma_y_2-1)
    sqnorm_mid = torch.clamp(torch.sum(mid**2, dim=-1, keepdim=True), 0, 1 - 1e-5)
    gamma_mid = 1/torch.sqrt(1-sqnorm_mid)
    midpoint = (gamma_mid/(1+gamma_mid))*mid


    return midpoint

# def cal_centroid(entities):
#
#     numerator = torch.zeros_like(entities[0])
#     denominator = 0
#     for entity in entities:
#         sqnorm_entity = torch.clamp(torch.sum(entity**2, dim=-1, keepdim=True), 0, 1 - 1e-5)
#         gamma_entity_square = 1 / (1 - sqnorm_entity)
#         numerator += gamma_entity_square*entity
#         denominator += (gamma_entity_square-0.5)
#
#     coaddition = numerator/denominator
#
#     sqnorm_coaddition = torch.clamp(torch.sum(coaddition**2, dim=-1, keepdim=True), 0, 1 - 1e-5)
#     gamma_coaddition = 1/torch.sqrt(1-sqnorm_coaddition)
#     centroid = (gamma_coaddition/(1+gamma_coaddition))*coaddition
#
#     A,B,C = entities[0], entities[1], entities[2]
#     AB = calculate_dist(A,B)
#     BC = calculate_dist(B,C)
#     CA = calculate_dist(C,A)
#     A2gamma = calculate_dist(centroid,A)
#     B2gamma = calculate_dist(centroid,B)
#     C2gamma = calculate_dist(centroid,C)
#     return -AB-BC-CA+2*(A2gamma+B2gamma+C2gamma)

def cat_rel(relations):
    compound = torch.zeros_like(relations[0])
    result = 0
    for rel in relations:
        
        result=p_sum(rel,compound)
        compound = result

def cat_rel(relations):
    compound = torch.zeros_like(relations[0])
    result = 0
    for rel in relations:
        
        result=p_sum(rel,compound)
        compound = result


    return compound

def cal_centroid_multi(entities):

    numerator = 0
    denominator = 0
    for entity in entities:
        sqnorm_entity = torch.clamp(torch.sum(entity**2, dim=-1, keepdim=True), 0, 1 - 1e-5)
        gamma_entity_square = 1 / (1 - sqnorm_entity)
        numerator += gamma_entity_square*entity
        denominator += (gamma_entity_square-0.5)

    coaddition = numerator/denominator

    sqnorm_coaddition = torch.clamp(torch.sum(coaddition*coaddition, dim=-1, keepdim=True), 0, 1 - 1e-5)
    gamma_coaddition = 1/torch.sqrt(1-sqnorm_coaddition)
    centroid = (gamma_coaddition/(1+gamma_coaddition))*coaddition


    return centroid

# def cal_square(data):
#     A, B, C = data[0], data[1], data[2]
#     AB = p_sum(-A, B)
#     CB = p_sum(-C, B)
#     CA = p_sum(-C, A)
#
#     Eucliden_sqdist_AB = torch.sum((A-B)**2, dim=-1, keepdim=True)
#     Eucliden_dist_AB = torch.norm(A - B, dim=-1, keepdim=True)
#
#     Eucliden_sqdist_BC = torch.sum((B - C) ** 2, dim=-1, keepdim=True)
#     Eucliden_dist_BC = torch.norm(B - C, dim=-1, keepdim=True)
#
#     Eucliden_sqdist_AC = torch.sum((C - A) ** 2, dim=-1, keepdim=True)
#     Eucliden_dist_AC = torch.norm(C - A, dim=-1, keepdim=True)
#
#     P_ABC = p_sum(Eucliden_sqdist_BC, Eucliden_dist_AC)
#     P_ABC = p_sum(-Eucliden_sqdist_AB,P_ABC)
#
#     R_AB = 2*Eucliden_dist_BC*Eucliden_sqdist_AC
#
#     Q_AB = (1+Eucliden_sqdist_BC)*(1+Eucliden_sqdist_AC)
#
#     cos_C = torch.clamp((P_ABC*R_AB/((1+P_ABC)*Q_AB)).squeeze(),0,1-1e-5)
#
#     sinC = torch.clamp(torch.sqrt(1-cos_C*cos_C),0,1-1e-5)
#
#     len_BC = cal_vector_length(p_sum(-B,C))
#     len_CA = cal_vector_length(p_sum(-C,A))


    # S_ABC = len_BC*len_CA*sinC*0.25    # len_AB = torch.tensor(cal_vector_length(AB))
    # len_AB = len_AB.unsqueeze(-1).repeat(1, 1, AB.shape[-1])

    # len_BC = torch.tensor(cal_vector_length(CB))
    # len_BC = cal_vector_length(CB)
    # # len_BC.clone().detach()
    # # len_BC  = cal_vector_length(CB)
    # # len_BC = len_BC.unsqueeze(-1).repeat(1, 1, CB.shape[-1])
    #
    # # len_CA = torch.tensor(cal_vector_length(CA))
    # len_CA = cal_vector_length(CA)
    # # len_CA.clone().detach()
    # # len_CA = cal_vector_length(CA)
    # # len_CA=len_CA.unsqueeze(-1).repeat(1,1,CA.shape[-1])
    #
    # # cosC = (CB/len_BC)*(CA/len_CA)
    # numerator =  torch.sum(CB*CA, dim=-1, keepdim=True).squeeze()
    # denominator = len_BC*len_CA
    # cosC = numerator/denominator
    # # cosC = torch.sum(CB*CA, dim=-1, keepdim=True).squeeze()
    # sinC = torch.sqrt(1-cosC**2)
    #
    # S_ABC = len_BC*len_CA*sinC*0.5*0.5
    # numerator = 1+torch.cosh(len_a)+torch.cosh(len_b)+torch.cosh(len_c)
    # denominator = 4*torch.cosh(len_a*0.5)*torch.cosh(len_b*0.5)*torch.cosh(len_c*0.5)
    # cos_half_square = numerator/denominator

    # return

# def cal_euclidean_square(data):
#     # a = C-B
#     # b = A-C
#     # c = B-A
#     numerator = torch.zeros_like(data[0])
#     denominator = 0
#     for entity in data:
#         sqnorm_entity = torch.clamp(torch.sum(entity ** 2, dim=-1, keepdim=True), 0, 1 - 1e-5)
#         gamma_entity_square = 1 / (1 - sqnorm_entity)
#         numerator += gamma_entity_square * entity
#         denominator += (gamma_entity_square - 0.5)
#
#     coaddition = numerator / denominator
#
#     sqnorm_coaddition = torch.clamp(torch.sum(coaddition * coaddition, dim=-1, keepdim=True), 0, 1 - 1e-5)
#     gamma_coaddition = 1 / torch.sqrt(1 - sqnorm_coaddition)
#     centroid = (gamma_coaddition / (1 + gamma_coaddition)) * coaddition
#
#     centroid2origin = calculate_dist(0, centroid)
#
#     A, B, C = data[0], data[1], data[2]
#     AB = p_sum(-A, B)
#     CB = p_sum(-C, B)
#     CA = p_sum(-C, A)
#
#     A, B, C = data[0], data[1], data[2]
#     AB = p_sum(-A, B)
#     CB = p_sum(-C, B)
#     CA = p_sum(-C, A)
#
#     norm_BC =  torch.clamp(torch.norm(CB, 2, dim=-1, keepdim=True), min=1e-10)
#     norm_CA =  torch.clamp(torch.norm(CA, 2, dim=-1, keepdim=True), min=1e-10)
#
#     cosC = (torch.sum(CB*CA, dim=-1, keepdim=True)/(norm_BC*norm_CA))
#     sinC = torch.sqrt(1-cosC*cosC)
#     S_ABC = (sinC*norm_BC*norm_CA*0.5/1.29975).squeeze()/centroid2origin
#
#
#     return S_ABC


def cal_perimeter(data):
    A, B, C = data[0], data[1], data[2]
    AB = p_sum(-A, B)
    BC = p_sum(-C, B)
    AC = p_sum(-C, A)

    return AB, BC, AC

def cal_h(data):
    A, B, C = data[0], data[1], data[2]
    AB = p_sum(-A, B)
    CB = p_sum(-C, B)
    AC = p_sum(-A, C)

    norm_AB = torch.clamp(torch.norm(AB, 2, dim=-1, keepdim=True), min=1e-10)
    norm_CB = torch.clamp(torch.norm(CB, 2, dim=-1, keepdim=True), min=1e-10)
    norm_CA = torch.clamp(torch.norm(AC, 2, dim=-1, keepdim=True), min=1e-10)

    # cosC = (torch.sum(CB * CA, dim=-1, keepdim=True) / (norm_CB * norm_CA))
    # sinC = torch.sqrt(1 - cosC * cosC)
    L_AC_M = cal_vector_length(AC)
    cosA = torch.sum(AB * AC, dim=-1, keepdim=True) / (norm_AB * norm_CA)
    sinA = torch.sqrt(1 - cosA * cosA).squeeze()
    H_AB_M = L_AC_M*sinA
    #
    #
    # L_AB_M = cal_vector_length(AB)
    # cosB = torch.sum(AB * CB, dim=-1, keepdim=True)/(norm_AB * norm_CB)
    # sinB = torch.sqrt(1 - cosB * cosB).squeeze()
    # H_BC_M = L_AB_M*sinB


    # L_BC_M = cal_vector_length(CB)
    # cosC = torch.sum(CB * AC, dim=-1, keepdim=True) / (norm_CB * norm_CA)
    # sinC = torch.sqrt(1 - cosC * cosC).squeeze()
    # H_AC_M = L_BC_M*sinC
    return H_AB_M**2

def cal_mean_square(data):
    # a = C-B
    # b = A-C
    # c = B-A
    numerator = torch.zeros_like(data[0])
    denominator = 0
    for entity in data:
        sqnorm_entity = torch.clamp(torch.sum(entity ** 2, dim=-1, keepdim=True), 0, 1 - 1e-5)
        gamma_entity_square = 1 / (1 - sqnorm_entity)
        numerator += gamma_entity_square * entity
        denominator += (gamma_entity_square - 0.5)

    coaddition = numerator / denominator

    sqnorm_coaddition = torch.clamp(torch.sum(coaddition * coaddition, dim=-1, keepdim=True), 0, 1 - 1e-5)
    gamma_coaddition = 1 / torch.sqrt(1 - sqnorm_coaddition)
    centroid = (gamma_coaddition / (1 + gamma_coaddition)) * coaddition

    # centroid2origin = calculate_dist(0, centroid)

    A, B, C = data[0], data[1], data[2]
    # AB = p_sum(-A, B)
    # CB = p_sum(-C, B)
    # CA = p_sum(-C, A)

    A2Centroid = calculate_dist(A,centroid)
    B2Centroid = calculate_dist(B,centroid)
    C2Centroid = calculate_dist(C,centroid)

    mean_square = (A2Centroid+B2Centroid+C2Centroid)



    return mean_square

def cal_distance_to_midpoint(data):
    A, B, C = data[0], data[1], data[2]
    midpoint = cal_midpoint(A,B)
    # AB = calculate_dist(A,B)
    CM = calculate_dist(midpoint,C)
    return CM


def cal_weighted_centroid_multi(entities):
    average_centroid = cal_centroid_multi(entities)
    weighted_centroid = 0

    weight = []

    sigma_weight = 0

    for entity in entities:
        exp_neg_dist = torch.exp(-torch.norm(entity - average_centroid, dim=-1)/10)
        sigma_weight += exp_neg_dist
        weight.append(exp_neg_dist)

    # numerator = 0
    # denominator = 0
    for i in range(len(entities)):
        weight_i = weight[i] / sigma_weight
        weight_i_reshape = weight_i.reshape(entities[0].shape[0], entities[0].shape[1], 1).repeat(1, 1,
                                                                                                  entities[0].shape[2])
        weighted_centroid += weight_i_reshape * entities[i]
    return weighted_centroid

    #     sqnorm_entity = torch.clamp(torch.sum(entities[i] ** 2, dim=-1, keepdim=True), 0, 1 - 1e-5)
    #     gamma_entity_square = 1 / (1 - sqnorm_entity)
    #     numerator += weight_i_reshape*gamma_entity_square * entities[i]
    #     denominator += weight_i_reshape*(gamma_entity_square - 0.5)
    #
    # coaddition = numerator / denominator
    # sqnorm_coaddition = torch.clamp(torch.sum(coaddition * coaddition, dim=-1, keepdim=True), 0, 1 - 1e-5)
    # gamma_coaddition = 1 / torch.sqrt(1 - sqnorm_coaddition)
    # centroid = (gamma_coaddition / (1 + gamma_coaddition)) * coaddition
    # return centroid

def cal_max_length(data,point_embbedings):
    arity = len(point_embbedings)
    dist = calculate_dist(point_embbedings[0],point_embbedings[1])
    for i in range(1,arity):
        for j in range(i):
            if i==1:
                continue

            len_i_j = calculate_dist(point_embbedings[i],point_embbedings[j])

            dist = torch.hstack((dist,len_i_j))
    dist = dist.reshape(data[0].shape[0],int((arity*arity-arity)/2),data[0].shape[1])
    dist = dist.permute(0,2,1)
    dist_sorted, _ = torch.sort(dist,dim=-1,descending=True)
    dist_max = dist_sorted[:,:,0]
    dist_max[:,0] = dist_sorted[:,0,-1]
    dist_sec = dist_sorted[:,:,1]
    return dist_max

def cal_all_length(data,point_embbedings):
    arity = len(point_embbedings)
    dist = calculate_dist(point_embbedings[0],point_embbedings[1])
    for i in range(1,arity):
        for j in range(i):
            if i==1:
                continue

            len_i_j = calculate_dist(point_embbedings[i],point_embbedings[j])

            dist = torch.hstack((dist,len_i_j))
    dist = dist.reshape(data[0].shape[0],int((arity*arity-arity)/2),data[0].shape[1])
    dist = dist.permute(0,2,1)
    dist_sorted, _ = torch.sort(dist,dim=-1,descending=True)
    dist_max = dist_sorted[:,:,0]
    dist_max[:,0] = dist_sorted[:,0,-1]
    dist_sec = dist_sorted[:,:,1]
    return dist_max

def cal_sum_length(data,point_embbedings):
    arity = len(point_embbedings)
    dist = 0
    for i in range(1,arity):
        for j in range(i):
            dist += calculate_dist(point_embbedings[i],point_embbedings[j])
    return dist


def cal_perimeter_n(point_embbedings):
    arity = len(point_embbedings)
    dist = 0
    for i in range(1,arity):
        for j in range(i):
            if abs(i - j) != 1 and abs(i - j) != arity-1:
                continue
            dist += calculate_dist(point_embbedings[i],point_embbedings[j])
    return dist

def cal_centroid_Eu(entities):
    numerator = 0
    denominator = 0
    for entity in entities:

        numerator +=  entity
        denominator += 1


    centroid =  numerator / denominator

    return centroid

def cal_weighted_centroid_Eu(entities):

    average_centroid = cal_centroid_Eu(entities)
    weighted_centroid = 0

    weight = []

    sigma_weight = 0

    for entity in entities:

        exp_neg_dist = torch.exp(-torch.norm(entity-average_centroid,dim=-1))
        sigma_weight += exp_neg_dist
        weight.append(exp_neg_dist)

    for i in range(len(entities)):
        weight_i = weight[i]/sigma_weight
        weight_i_reshape = weight_i.reshape(entities[0].shape[0],entities[0].shape[1],1).repeat(1,1,entities[0].shape[2])
        weighted_centroid += weight_i_reshape*entities[i]

    return weighted_centroid

if  __name__ == '__main__':

    # A_2 =torch.Tensor([0.60966])
    # B_2 = torch.Tensor([0.48908])
    # C_2 = torch.Tensor([0.53018])

    # mid = p_sum(A_2,B_2)
    # mid = p_sum(-C_2,mid)
    # print(mid)
    # print(1.733*3/4)

    # A = torch.FloatTensor([[1,2,3,4,6],[6,7,8,9,11],[10,3,6,8,4],[1,2,3,4,6]])
    # B = torch.FloatTensor([[6,7,8,9,10],[1,2,3,4,5],[1,8,3,4,5],[1,2,3,4,5]])
    # C = torch.FloatTensor([[10,3,6,8,4],[1,8,3,4,5],[6,7,8,9,11],[6,7,8,9,11]])
    # D = torch.hstack((A,B))
    # D =torch.hstack((D,C))
    # print(D.resize(4,3,5))
    # D = D.resize(4,3,5)
    # D = torch.mean(D,dim=1)
    # print(D)

    # A = torch.randn(4,5)
    # B = torch.randn(4,5)
    # C = torch.randn(4,5)
    # E = torch.hstack((A,B,C))
    # E = E.resize(4,3,5)
    # E = torch.mean(E,dim=1)
    # print(E)

    # a= [14,10,14]
    # print(a[:2])

    # a = torch.ones(0,10)
    # c = torch.ones((1,10)).detach()
    # d = torch.cat((a,c),0)
    # f = torch.ones((1,10)).detach()
    # e = torch.cat((d,f),0)
    # print(d,e)
    import numpy as np
    a = torch.zeros(10)
    index = [1,3,5]
    idx = torch.from_numpy(np.array(index))
    a[idx]=1
    print(a)

