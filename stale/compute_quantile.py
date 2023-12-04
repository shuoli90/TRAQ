import os
import json
import numpy as np
import argparse
import MHT
from skopt.space import Real
from skopt import gp_minimize
from skopt.utils import use_named_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.1)
    # parser.add_argument('--input', type=str, required=True)
    # parser.add_argument('--all_scores', type=str,
    #                     default='all_scores_cosine_new.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    random_seed = args.seed

    # read in data files
    with open(os.path.join(
            'collected',
            'true_scores_cosine_match.json'), "r") as src_file:
        true_scores_cosine = json.load(src_file)
    with open(os.path.join(
            'collected',
            'most_relevant_cosine_match.json'), "r") as src_file:
        most_relevant_cosine = json.load(src_file)
    with open(os.path.join(
            'collected',
            'retrieve_scores_cosine_match.json'), "r") as src_file:
        retrieve_scores_cosine = json.load(src_file)
    with open(os.path.join(
            'collected',
            'all_scores_cosine_match.json'), "r") as src_file:
        all_scores_cosine = json.load(src_file)

    with open(os.path.join(
            'collected',
            'all_scores_cosine_match.json'), "r") as src_file:
        dpr_records = json.load(src_file)

    # split indices into calibration and test
    alpha = args.alpha
    np.random.seed(random_seed)
    length = np.arange(len(dpr_records))
    np.random.shuffle(length)
    calibration_indices = length[:int(len(dpr_records) * 0.5)].tolist()
    test_indices = length[int(len(dpr_records) * 0.5):].tolist()
    calibration = [dpr_records[i] for i in calibration_indices]
    test = [dpr_records[i] for i in test_indices]

    true_scores_cosine_cal = [np.array(true_scores_cosine[i])
                              for i in calibration_indices]
    true_scores_cosine_test = [np.array(true_scores_cosine[i])
                               for i in test_indices]
    most_relevant_cosine_cal = [np.array(most_relevant_cosine[i])
                                for i in calibration_indices]
    most_relevant_cosine_test = [np.array(most_relevant_cosine[i])
                                 for i in test_indices]
    retrieve_scores_cosine_cal = [np.array(retrieve_scores_cosine[i])
                                  for i in calibration_indices]
    retrieve_scores_cosine_test = [np.array(retrieve_scores_cosine[i])
                                   for i in test_indices]
    all_scores_cosine_cal = [np.array(all_scores_cosine[i])
                             for i in calibration_indices]
    all_scores_cosine_test = [np.array(all_scores_cosine[i])
                              for i in test_indices]
    true_scores_cosine_cal = np.hstack(true_scores_cosine_cal)
    true_scores_cosine_test = np.hstack(true_scores_cosine_test)
    most_relevant_cosine_cal = np.hstack(most_relevant_cosine_cal)
    most_relevant_cosine_test = np.hstack(most_relevant_cosine_test)
    retrieve_scores_cosine_cal = np.hstack(retrieve_scores_cosine_cal)
    retrieve_scores_cosine_test = np.hstack(retrieve_scores_cosine_test)
    all_scores_cosine_cal = np.stack(all_scores_cosine_cal)
    all_scores_cosine_test = np.stack(all_scores_cosine_test)

    # compute quantiles
    thr_most_relevant = np.quantile(most_relevant_cosine_cal, alpha)
    print('coverage', np.mean(most_relevant_cosine_test >= thr_most_relevant))
    print('size', np.mean(np.sum(all_scores_cosine_test >= thr_most_relevant, 1)))

    # split test set into parameter-training and test
    # true_score
    indices = np.arange(len(true_scores_cosine_test))
    np.random.shuffle(indices)
    cal_indices = indices[:int(len(true_scores_cosine_test) * 0.5)]
    test_indices = indices[int(len(true_scores_cosine_test) * 0.5):]
    true_scores_cosine_param = true_scores_cosine_test[cal_indices]
    true_scores_cosine_test = true_scores_cosine_test[test_indices]

    # most_relevant
    indices = np.arange(len(most_relevant_cosine_test))
    np.random.shuffle(indices)
    cal_indices = indices[:int(len(most_relevant_cosine_test) * 0.5)]
    test_indices = indices[int(len(most_relevant_cosine_test) * 0.5):]
    most_relevant_cosine_param = most_relevant_cosine_test[cal_indices]
    most_relevant_cosine_test = most_relevant_cosine_test[test_indices]

    # retrieve_scores
    indices = np.arange(len(retrieve_scores_cosine_test))
    np.random.shuffle(indices)
    cal_indices = indices[:int(len(retrieve_scores_cosine_test) * 0.5)]
    test_indices = indices[int(len(retrieve_scores_cosine_test) * 0.5):]
    retrieve_scores_cosine_param = retrieve_scores_cosine_test[cal_indices]
    retrieve_scores_cosine_test = retrieve_scores_cosine_test[test_indices]

    # all_scores
    indices = np.arange(len(all_scores_cosine_test))
    np.random.shuffle(indices)
    cal_indices = indices[:int(len(all_scores_cosine_test) * 0.5)]
    test_indices = indices[int(len(all_scores_cosine_test) * 0.5):]
    all_scores_cosine_param = all_scores_cosine_test[cal_indices]
    all_scores_cosine_test = all_scores_cosine_test[test_indices]

    # compute p-values for retrieval
    true_pvalues = MHT.cal_pvalues(true_scores_cosine_cal,
                                   true_scores_cosine_test)
    most_relevant_pvalues = MHT.cal_pvalues(most_relevant_cosine_cal,
                                            most_relevant_cosine_test)
    retrieve_pvalues = MHT.cal_pvalues(retrieve_scores_cosine_cal,
                                       retrieve_scores_cosine_test)

    # generate dummy p-values
    true_dummy_cal = np.random.uniform(size=true_scores_cosine_cal.shape)

    # generate dummy p-values for question answering
    true_dummy_param = np.random.uniform(
        size=true_scores_cosine_param.shape)
    true_dummy_test = np.random.uniform(
        size=true_scores_cosine_test.shape)
    most_relevant_dummy_param = np.random.uniform(
        size=most_relevant_cosine_param.shape)
    most_relevant_dummy_test = np.random.uniform(
        size=most_relevant_cosine_test.shape)
    retrieve_dummy_param = np.random.uniform(
        size=retrieve_scores_cosine_param.shape)
    retrieve_dummy_test = np.random.uniform(
        size=retrieve_scores_cosine_test.shape)
    all_dummy_param = np.random.uniform(
        size=all_scores_cosine_param.shape)
    all_dummy_test = np.random.uniform(
        size=all_scores_cosine_test.shape)

    # compute MHT pvalues
    # hmp = MHT.harmonic_mean_p_value(p_values=[true_pvalues,
    #                                           true_dummy_pvalues])
    """
    Weight HMP module
    """
    w1 = Real(name='w1', low=0.0, high=1.0)
    w2 = Real(name='w2', low=0.0, high=1.0)

    def softmax(vec):
        nom = np.exp(vec - np.mean(vec))
        return nom / np.sum(nom)

    # Gather the search-space dimensions in a list.
    dimensions = [w1, w2]
    As_train = true_scores_cosine_cal
    Bs_train = true_dummy_cal
    # As_confidences = true_scores_cosine_param
    As_confidences = np.random.uniform(
        size=all_scores_cosine_param.shape)
    Bs_confidences = all_dummy_param
    # As_test = true_scores_cosine_test
    As_test = np.random.uniform(
        size=all_scores_cosine_test.shape)
    Bs_test = all_dummy_test
    epsilon_HMP = 0.05

    @use_named_args(dimensions=dimensions)
    def my_objective_function(w1, w2):
        weights = softmax(np.array([w1, w2])).reshape(-1, 1)
        includes = np.zeros(As_confidences.shape[0])
        for i in range(As_confidences.shape[1]):
            p_values = [MHT.cal_pvalues(cals=As_train,
                                        scores=As_confidences[:, i]),
                        MHT.cal_pvalues(cals=Bs_train,
                                        scores=Bs_confidences[:, i])]
            p_values = np.stack(p_values)
            harmonic_p_values = MHT.harmonic_mean_p_value(
                p_values,
                weights=np.tile(weights, (1, p_values.shape[1])))
            includes += harmonic_p_values >= epsilon_HMP
        return np.mean(includes)

    def HMP_eval(w1, w2):
        weights = softmax(np.array([w1, w2])).reshape(-1, 1)
        includes = np.zeros(As_test.shape[0])
        for i in range(As_test.shape[1]):
            p_values = [MHT.cal_pvalues(cals=As_train, scores=As_test[:, i]),
                        MHT.cal_pvalues(cals=Bs_train, scores=Bs_test[:, i])]
            p_values = np.stack(p_values)
            harmonic_p_values = MHT.harmonic_mean_p_value(
                p_values,
                weights=np.tile(weights, (1, p_values.shape[1])))
            includes += harmonic_p_values >= epsilon_HMP
        return np.mean(includes), includes

    result = gp_minimize(func=my_objective_function,
                         dimensions=dimensions,
                         acq_func="EI",      # the acquisition function
                         n_calls=10,
                         random_state=random_seed,
                         verbose=True)

    print("Best fitness:", result.fun)
    print("Best parameters:", softmax(result.x))

    size, coverage = HMP_eval(w1=result.x[0], w2=result.x[1])
                              
