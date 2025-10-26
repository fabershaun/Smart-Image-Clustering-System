import pickle
import pandas as pd
from sklearn.metrics import rand_score


def evaluate_clustering_result(actual_clusters_filename, clustering_result):
    """
    evaluating clustering results: algo outcome vs annotated clusters
    :param actual_clusters_filename: filename with the annotated clustering outcome
    :param clustering_result: clustering algo outcome

    """

    mem2cluster = dict()
    # create a member2cluster dictionary
    for c, members in clustering_result.items():
        mem2cluster.update({m: c for m in members})

    actual = pd.read_csv(actual_clusters_filename)
    actual = actual.drop(actual[actual['cluster'] == -1].index)
    print(f'clustered images in labeled data: {len(actual)}')
    actual = dict(zip(actual['filename'], actual['cluster']))

    # find the number of members in common in the two solutions
    clst_solution = set(mem2cluster.values()); clst_actual = set(actual.values())
    print(f'clusters in solution: {len(clst_solution)} and actual: {len(clst_actual)}')
    members_in_common = sorted(list(set(mem2cluster.keys()).intersection(set(actual.keys()))))
    print(f'clustered in solution: {len(mem2cluster)} and actual: {len(actual)}, members in common: {len(members_in_common)}')

    actual_member_cls = [actual[id] for id in members_in_common]
    cfound_member_cls = [mem2cluster[id] for id in members_in_common]
    print(f'rand score for {len(cfound_member_cls)} members common in clustering result and the actual grouping: '
          f'{round(rand_score(cfound_member_cls, actual_member_cls), 4)}')


def load_from_pickle_example():
    with open('data/flowers/image-features.pkl', 'rb') as fin:
        dataset = pickle.load(fin)

    print(type(dataset), len(dataset))
    for k, v in dataset.items():
        print(k, v)


if __name__ == '__main__':
    load_from_pickle_example()
