import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch as torch
import  argparse
from model import *
from load_data import *
from main_together_copy  import  Experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="LeProhiALL-LeCentroid", nargs="?",
                        help="Which dataset to test")
    parser.add_argument("--model", type=str, default="LeCentroid", nargs="?",
                        help="Which model to use: poincare or euclidean.")
    parser.add_argument("--num_iterations", type=int, default=510, nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?",
                        help="Batch size.")
    parser.add_argument("--nneg", type=int, default=100, nargs="?",
                        help="Number of negative samples.")
    parser.add_argument("--lr", type=float, default=2, nargs="?",
                        help="Learning rate.")
    parser.add_argument("--dim", type=int, default=512, nargs="?",
                        help="Embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True, nargs="?",
                        help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--test_batch_size", type=int, default=1, nargs="?",
                        help="Test Batch Size.")

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True
    seed = 40
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    d = Data_whole_and_split(data_dir=data_dir)

    experiment = Experiment(learning_rate=args.lr, batch_size=args.batch_size,
                            num_iterations=args.num_iterations, dim=args.dim,
                            cuda=args.cuda, nneg=args.nneg, model=args.model,test_batch_size=args.test_batch_size)

    experiment.entity2idxs = {d.entities[i]: i for i in range(len(d.entities))}
    experiment.idxs2entity = {i: d.entities[i] for i in range(len(d.entities))}
    experiment.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}
    experiment.idxs2relation = {i: d.relations[i] for i in range(len(d.relations))}

    # experiment.primary_entities_indexes = [experiment.entity2idxs[item] for item in d.primary_entities]
    # experiment.affiliated_entities_indexes = [experiment.entity2idxs[item] for item in d.affilated_entities]
    all_fact = experiment.get_data_idxs(d.data)
    if args.model == "E_Centroid":
        model = E_centroid(d, args.dim)
    elif args.model == "Centroid":
        model = Centroid(d,args.dim)
        model.load_state_dict(torch.load('./data/'+args.dataset+'/checkpoint_model_epoch_400.pth.tar'))
        # experiment.evaluate_batch_ckpt(model,d)

        
        experiment.evaluate_for_checkpoint(model, d)
    elif args.model == "LeCentroid":
        model = LeCentroid(d,args.dim)
        model.load_state_dict(torch.load('./data/'+args.dataset+'/checkpoint_model_epoch_400.pth.tar'))
        experiment.evaluate_batch_ckpt(model,d)

        
        # experiment.evaluate_for_checkpoint(model, d)

    elif args.model == "poincare":
        model = MuRP(d,args.dim)
        model.load_state_dict(torch.load('./data/'+args.dataset+'/checkpoint_model_epoch_808.pth.tar'))

        experiment.evaluate_for_checkpoint(model, d)

    # model.load_state_dict(torch.load('./data/'+args.dataset+'/checkpoint_model_epoch_400.pth.tar'))

    # experiment.evaluate_batch_ckpt(model,d)

    # experiment.find_KNN(model,d)
    # experiment.evaluate(model,d)


