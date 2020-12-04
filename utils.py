import os
import json
import glob
import pandas as pd


TRAINING_DATA_PATH = 'training_data'
TOP_CLASS_SPLITS_PATH = os.path.join(TRAINING_DATA_PATH, 'top_class_splits')
CITATION_SPLITS_PATH = os.path.join(TRAINING_DATA_PATH, 'citation_splits')
AUXILIARY_DAT_PATH = 'auxiliary_data'
MESH_DATA_PATH = os.path.join(AUXILIARY_DAT_PATH, 'mesh')


def join_splits(xs, ys):
    all_x = list()
    all_y = list()
    for x, y in zip(xs, ys):
        all_x += x
        all_y += y
    assert len(all_x) == len(all_y)
    return all_x, all_y


def load_from_split(split_path, split, feature_data, include_pids):
    ids = {x['paper_id']: x['label'] for x in json.load(open(os.path.join(split_path, f'{split}.json')))}
    id_list = list(ids.keys())
    x, y = list(zip(*[(feature_data[idx], ids[idx]) for idx in id_list]))
    if include_pids:
        return id_list, x, y
    else:
        return x, y


def load_top_class_embeddings_split(split, include_pids=False):
    assert split in ['train', 'val', 'test'], AssertionError(f'Unknown split name: {split}. Must choose train, val, '
                                                             f'or test')
    embeddings = load_paper_id_2_embeddings()
    return load_from_split(TOP_CLASS_SPLITS_PATH, split, embeddings, include_pids)


def load_citation_embeddings_split(split, include_pids=False):
    assert split in ['train', 'val', 'test'], AssertionError(f'Unknown split name: {split}. Must choose train, val, '
                                                             f'or test')
    embeddings = load_paper_id_2_embeddings()
    return load_from_split(CITATION_SPLITS_PATH, split, embeddings, include_pids)


def load_top_class_embeddings_with_c1_split(split, include_pids=False):
    assert split in ['train', 'val', 'test'], AssertionError(f'Unknown split name: {split}. Must choose train, val, '
                                                             f'or test')
    embeddings = load_paper_id_2_embeddings()
    abstracts = json.load(open(os.path.join(TOP_CLASS_SPLITS_PATH, f'{split}.json')))
    ids = {x['paper_id']: x['label'] for x in abstracts}
    c1 = {x['paper_id']: float(x['1_year_count']) for x in abstracts}
    id_list = list(ids.keys())
    x, y = list(zip(*[([*embeddings[idx], c1[idx]], ids[idx]) for idx in id_list]))
    if include_pids:
        return id_list, x, y
    else:
        return x, y


def load_citation_embeddings_with_c1_split(split, include_pids=False):
    assert split in ['train', 'val', 'test'], AssertionError(f'Unknown split name: {split}. Must choose train, val, '
                                                             f'or test')
    embeddings = load_paper_id_2_embeddings()
    abstracts = json.load(open(os.path.join(CITATION_SPLITS_PATH, f'{split}.json')))
    ids = {x['paper_id']: x['label'] for x in abstracts}
    c1 = {x['paper_id']: float(x['1_year_count']) for x in abstracts}
    id_list = list(ids.keys())
    x, y = list(zip(*[([*embeddings[idx], c1[idx]], ids[idx]) for idx in id_list]))
    if include_pids:
        return id_list, x, y
    else:
        return x, y


def load_top_class_abstracts_split(split):
    assert split in ['train', 'val', 'test'], AssertionError(f'Unknown split name: {split}. Must choose train, val, '
                                                             f'or test')
    return json.load(open(os.path.join(TOP_CLASS_SPLITS_PATH, f'{split}.json')))


def load_citation_abstracts_split(split):
    assert split in ['train', 'val', 'test'], AssertionError(f'Unknown split name: {split}. Must choose train, val, '
                                                             f'or test')
    return json.load(open(os.path.join(CITATION_SPLITS_PATH, f'{split}.json')))


def load_top_class_h_index_with_c1_split(split, include_pids=False):
    assert split in ['train', 'val', 'test'], AssertionError(f'Unknown split name: {split}. Must choose train, val, '
                                                             f'or test', include_pids)
    h_index_features = load_paper_id_2_h_index_features_with_c1()
    return load_from_split(TOP_CLASS_SPLITS_PATH, split, h_index_features, include_pids)


def load_top_class_h_index_split(split, include_pids=False):
    assert split in ['train', 'val', 'test'], AssertionError(f'Unknown split name: {split}. Must choose train, val, '
                                                             f'or test')
    h_index_features = load_paper_id_2_h_index_features()
    return load_from_split(TOP_CLASS_SPLITS_PATH, split, h_index_features, include_pids)


def load_citation_h_index_with_c1_split(split, include_pids=False):
    assert split in ['train', 'val', 'test'], AssertionError(f'Unknown split name: {split}. Must choose train, val, '
                                                             f'or test')
    h_index_features = load_paper_id_2_h_index_features_with_c1()
    return load_from_split(CITATION_SPLITS_PATH, split, h_index_features, include_pids)


def load_citation_h_index_split(split, include_pids=False):
    assert split in ['train', 'val', 'test'], AssertionError(f'Unknown split name: {split}. Must choose train, val, '
                                                             f'or test')
    h_index_features = load_paper_id_2_h_index_features()
    return load_from_split(CITATION_SPLITS_PATH, split, h_index_features, include_pids)


def load_paper_id_2_embeddings():
    return {x['paper_id']: x['embedding'] for x in [json.loads(x) for x in open(os.path.join(AUXILIARY_DAT_PATH,
                                                                                             'mesh_embeddings.jsonl'))]}


def load_paper_id_2_h_index_features():
    return json.load(open(os.path.join(AUXILIARY_DAT_PATH, 'paper_id_2_h_index_features.json')))


def load_paper_id_2_h_index_features_with_c1():
    return json.load(open(os.path.join(AUXILIARY_DAT_PATH, 'paper_id_2_h_index_features_with_c1.json')))


def load_paper_id_2_abstracts():
    return json.load(open(os.path.join(AUXILIARY_DAT_PATH, 'paper_id_2_abstracts.json')))


def load_paper_2_mesh_labels():
    data_files = glob.glob(os.path.join(MESH_DATA_PATH, '*.csv'))
    mesh_ids_with_label = dict(pd.concat([pd.read_csv(file) for file in data_files]).values.tolist())
    return mesh_ids_with_label


def load_paper_id_2_citation_years():
    return json.load(open(os.path.join(AUXILIARY_DAT_PATH, 'paper_id_2_citation_years.json')))


def load_paper_id_2_citation_counts():
    return json.load(open(os.path.join(AUXILIARY_DAT_PATH, 'paper_id_2_citation_counts.json')))


def load_author_year_2_h_index():
    return json.load(open(os.path.join(AUXILIARY_DAT_PATH, 'author_year_2_h_index.json')))


def load_paper_id_2_top_class():
    return json.load(open(os.path.join(AUXILIARY_DAT_PATH, 'paper_id_2_top_class.json')))


def load_paper_id_2_10_year_citation_count():
    return json.load(open(os.path.join(AUXILIARY_DAT_PATH, 'paper_id_2_10_year_citation_count.json')))


def load_paper_id_2_1_year_citation_count():
    return json.load(open(os.path.join(AUXILIARY_DAT_PATH, 'paper_id_2_1_year_citation_count.json')))


def load_paper_metadata_mag_mesh():
    return json.load(open(os.path.join(AUXILIARY_DAT_PATH, 'paper_metadata_mag_mesh.json')))
