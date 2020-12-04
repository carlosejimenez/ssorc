import os
import utils
import json
import numpy as np
from collections import defaultdict


def get_1_year_citations(citations, year):
    valid_keys = list(filter(lambda x: x != 'null' and int(x) <= year + 1, citations.keys()))
    return float(sum([citations[key] for key in valid_keys]))


def get_one_hot(category):
    vec = np.zeros(11, dtype=float)
    vec[category] = 1
    return vec.tolist()


def gen_paper_id_top_class():
    paper_id_2_citation_years = utils.load_paper_id_2_citation_years()
    mesh_metadata = utils.load_paper_metadata_mag_mesh()
    label_data = utils.load_paper_2_mesh_labels()
    year_mesh_label_2_pids = defaultdict(set)
    for pid, entry in mesh_metadata.items():
        if pid in label_data and entry['year'] is not None and pid in paper_id_2_citation_years:
            year_mesh_label_2_pids[(entry['year'], label_data[pid])].add(pid)
    paper_id_2_top_class = dict()
    for (year, mesh_label), papers in year_mesh_label_2_pids.items():
        num_citations = sorted([(sum(paper_id_2_citation_years[x].values()), x) for x in papers])
        if len(num_citations) < 50:
            continue
        top_10_per = {x[1] for x in num_citations[-round(len(num_citations) / 10):]}
        bottom_90_per = {x[1] for x in num_citations[:-round(len(num_citations) / 10)]}
        assert len(top_10_per.intersection(bottom_90_per)) == 0 and len(top_10_per) + len(bottom_90_per) == len(
            num_citations)
        for paper_id in top_10_per:
            paper_id_2_top_class[paper_id] = 1
        for paper_id in bottom_90_per:
            paper_id_2_top_class[paper_id] = 0
    json.dump(paper_id_2_top_class, open(os.path.join(utils.AUXILIARY_DAT_PATH, 'paper_id_2_top_class.json'), 'w+'))


def gen_paper_id_2_10_year_citation_counts():
    paper_id_2_citation_counts = utils.load_paper_id_2_citation_years()
    metadata = utils.load_paper_metadata_mag_mesh()
    label_data = utils.load_paper_2_mesh_labels()
    paper_id_2_10_year_count = dict()
    for pid, entry in metadata.items():
        if entry['year'] is not None and entry['year'] <= 2010:
            if pid in label_data and pid in paper_id_2_citation_counts:
                total_citations = sum(dict(filter(lambda x: x[0] != 'null' and int(x[0]) <= metadata[pid]['year'] + 10,
                                                  paper_id_2_citation_counts[pid].items())).values())
                paper_id_2_10_year_count[pid] = total_citations
    json.dump(paper_id_2_10_year_count, open(os.path.join(utils.AUXILIARY_DAT_PATH,
                                                          'paper_id_2_10_year_citation_count.json'), 'w+'))


def gen_h_index_features():
    labelled_paper_ids = utils.load_paper_id_2_abstracts()
    paper_id_2_mesh_label = utils.load_paper_2_mesh_labels()

    author_year_2_h_index = defaultdict(int)
    author_year_2_h_index.update(utils.load_author_year_2_h_index())
    paper_id_2_citation_years = utils.load_paper_id_2_citation_years()
    mesh_metadata = utils.load_paper_metadata_mag_mesh()

    paper_id_2_h_index_features = dict()
    paper_id_2_h_index_features_with_c1 = dict()
    for paper_id in labelled_paper_ids:
        metadata = mesh_metadata[paper_id]
        year = metadata['year']
        h_indices = list()
        for author in metadata['authors']:
            key = '_'.join([author, str(year)])
            h_indices.append(author_year_2_h_index[key])
        if len(h_indices) == 0:
            continue
        features = [
            np.min(h_indices),
            np.mean(h_indices),
            np.median(h_indices),
            np.max(h_indices),
            np.max(h_indices) - np.min(h_indices),
            *get_one_hot(paper_id_2_mesh_label[paper_id]),
        ]
        paper_id_2_h_index_features[paper_id] = list(map(float, features))
        paper_id_2_h_index_features_with_c1[paper_id] = [*list(map(float, features)),
                                                         get_1_year_citations(paper_id_2_citation_years[paper_id],
                                                                              year)]
    json.dump(paper_id_2_h_index_features, open(os.path.join(utils.AUXILIARY_DAT_PATH,
                                                             'paper_id_2_h_index_features.json'), 'w+'))
    json.dump(paper_id_2_h_index_features_with_c1,
              open(os.path.join(utils.AUXILIARY_DAT_PATH, 'paper_id_2_h_index_features_with_c1.json'), 'w+'))


def gen_top_class_splits():
    paper_id_2_top_class = utils.load_paper_id_2_top_class()
    paper_id_2_abstract = utils.load_paper_id_2_abstracts()
    paper_id_2_h_index_features = utils.load_paper_id_2_h_index_features()
    paper_id_2_1_year_count = utils.load_paper_id_2_1_year_citation_count()
    mesh_metadata = utils.load_paper_metadata_mag_mesh()
    train = list()
    val = list()
    test = list()
    missing = list()
    valid_ids = set(paper_id_2_top_class.keys())
    valid_ids &= set(paper_id_2_abstract.keys())
    valid_ids &= set(paper_id_2_h_index_features.keys())
    for paper_id in valid_ids:
        assert paper_id in mesh_metadata
        year = mesh_metadata[paper_id]['year']
        entry = {'paper_id': paper_id, 'text': paper_id_2_abstract[paper_id],
                 '1_year_count': paper_id_2_1_year_count[paper_id],
                 'label': paper_id_2_top_class[paper_id]}
        if year <= 2008:
            train.append(entry)
        elif year == 2009:
            val.append(entry)
        elif year in [2010]:
            test.append(entry)
        else:
            missing.append(missing)
    os.makedirs(utils.TOP_CLASS_SPLITS_PATH, exist_ok=True)
    json.dump(train, open(os.path.join(utils.TOP_CLASS_SPLITS_PATH, 'train.json'), 'w+'))
    json.dump(val, open(os.path.join(utils.TOP_CLASS_SPLITS_PATH, 'val.json'), 'w+'))
    json.dump(test, open(os.path.join(utils.TOP_CLASS_SPLITS_PATH, 'test.json'), 'w+'))


def gen_citation_splits():
    paper_id_10_year_citation_count = utils.load_paper_id_2_10_year_citation_count()
    paper_id_2_abstract = utils.load_paper_id_2_abstracts()
    paper_id_2_h_index_features = utils.load_paper_id_2_h_index_features()
    paper_id_2_1_year_count = utils.load_paper_id_2_1_year_citation_count()
    mesh_metadata = utils.load_paper_metadata_mag_mesh()
    train = list()
    val = list()
    test = list()
    missing = list()
    valid_ids = set(paper_id_10_year_citation_count.keys())
    valid_ids &= set(paper_id_2_abstract.keys())
    valid_ids &= set(paper_id_2_h_index_features.keys())
    for paper_id in valid_ids:
        assert paper_id in mesh_metadata
        year = mesh_metadata[paper_id]['year']
        entry = {'paper_id': paper_id, 'text': paper_id_2_abstract[paper_id],
                 '1_year_count': paper_id_2_1_year_count[paper_id],
                 'label': paper_id_10_year_citation_count[paper_id]}
        if year <= 2008:
            train.append(entry)
        elif year == 2009:
            val.append(entry)
        elif year in [2010]:
            test.append(entry)
        else:
            missing.append(missing)
    os.makedirs(utils.CITATION_SPLITS_PATH, exist_ok=True)
    json.dump(train, open(os.path.join(utils.CITATION_SPLITS_PATH, 'train.json'), 'w+'))
    json.dump(val, open(os.path.join(utils.CITATION_SPLITS_PATH, 'val.json'), 'w+'))
    json.dump(test, open(os.path.join(utils.CITATION_SPLITS_PATH, 'test.json'), 'w+'))


if __name__ == '__main__':
    print('Generating top class labels')
    gen_paper_id_top_class()
    print('Generating 10 year citation labels')
    gen_paper_id_2_10_year_citation_counts()
    print('Generating h-index features')
    gen_h_index_features()
    print('Generating top-class splits')
    gen_top_class_splits()
    print('Generating citation count splits')
    gen_citation_splits()
