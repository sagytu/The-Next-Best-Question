import argparse
from process_dataset import read_preprocess
from models_creation import train_model
from utils import mask_data, save_results
from TNBQ import the_next_best_question, global_shap_baseline, the_next_best_question_weighted,\
                    the_next_best_question_radius_neighborhood


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help="The dataset name")
parser.add_argument('--method', type=str, default='TNBQ', required=False, help="Which method / extension to use")
parser.add_argument('--parameters','--list', type=float, default=[15], nargs='+', required=False, help="Parameters of the selected method")
parser.add_argument('--seed', type=int, default=10, required=False, help="Random seed")
args = parser.parse_args()


if __name__ == '__main__':

    print('Read and preprocess')
    X_train, X_test, y_train, y_test = read_preprocess(args.dataset)

    print('Train Model')
    model = train_model(args.dataset, X_train, y_train, _seed=args.seed)

    print('Mask data')
    X_test_masked, mask_data_filling = mask_data(X_test, X_train.shape[1], args.seed)

    if args.method == 'TNBQ':
        print('The Next Best Question')
        K = int(args.parameters[0])
        res_lst = the_next_best_question(X_train, X_test_masked, y_test, model, X_train.shape[1], mask_data_filling, K)
        save_results(res_lst, "The Next Best Question", args.parameters, args.seed, args.dataset.title())

    if args.method.lower() == 'global':
        print('Global Feature Importance Acquisition')
        res_lst = global_shap_baseline(X_train, X_test_masked, y_test, model, X_train.shape[1], mask_data_filling)
        save_results(res_lst, "Global Feature Importance Acquisition", args.parameters, args.seed, args.dataset.title())

    if args.method.lower() == 'weighted':
        print('The Next Best Question - Weighted KNN Extension')
        K = int(args.parameters[0])
        scale_range = (args.parameters[1], args.parameters[2])
        print(K)
        print(scale_range)
        res_lst = the_next_best_question_weighted(X_train, X_test_masked, y_test, model, X_train.shape[1],
                                                  mask_data_filling, K, scale_range=scale_range)
        save_results(res_lst, "The Next Best Question - Weighted KNN Extension", args.parameters, args.seed, args.dataset.title())

    if args.method.lower() == 'radius':
        print('The Next Best Question - Radius Neighborhood Extension')
        dist_th = args.parameters[0]
        min_neigh_size = int(args.parameters[1])
        max_neigh_size = int(args.parameters[2])
        scale_range = (args.parameters[1], args.parameters[2])
        print(dist_th)
        print(min_neigh_size, max_neigh_size)
        res_lst = the_next_best_question_radius_neighborhood(X_train, X_test_masked, y_test, model, X_train.shape[1], mask_data_filling,
                                               dist_th=dist_th, min_neigh_size=min_neigh_size, max_neigh_size=max_neigh_size)
        save_results(res_lst, "The Next Best Question - Radius Neighborhoos Extension", args.parameters, args.seed, args.dataset.title())




