import numpy as np
import torch
import torch.utils.data
from util.data_util import pad_seq, pad_char_seq, pad_video_seq
import random

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_features, audio_features, mode='test'):
        super(Dataset, self).__init__()
        self.dataset = dataset
        
        for key in video_features.keys():
            if len(video_features[key].shape) == 4:
                video_features[key] = video_features[key].squeeze()
        
        self.video_features = video_features
        self.audio_features = audio_features

    def __getitem__(self, index):
        record = self.dataset[index]
        video_feature = self.video_features[record['vid']]
        audio_feature = self.audio_features[record['vid']]
        s_ind, e_ind = int(record['s_ind']), int(record['e_ind'])
        word_ids, char_ids = record['w_ids'], record['c_ids']
        return record, video_feature, audio_feature, word_ids, char_ids, s_ind, e_ind

    def __len__(self):
        return len(self.dataset)

def train_collate_fn(data):
    
    records, video_features, audio_features, word_ids, char_ids, s_inds, e_inds = zip(*data)
    # process word ids
    word_ids, _ = pad_seq(word_ids)
    word_ids = np.asarray(word_ids, dtype=np.int32)
    # process char ids
    char_ids, _ = pad_char_seq(char_ids)
    char_ids = np.asarray(char_ids, dtype=np.int32)
    # process video features
    vfeats, vfeat_lens = pad_video_seq(video_features)
    vfeats = np.asarray(vfeats, dtype=np.float32)
    vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)
    # process audio features
    afeats, afeat_lens = pad_video_seq(audio_features)
    afeats = np.asarray(afeats, dtype=np.float32)
    afeat_lens = np.asarray(afeat_lens, dtype=np.int32)  
    # process labels
    max_len = np.max(vfeat_lens)
    batch_size = vfeat_lens.shape[0]
    s_labels = np.asarray(s_inds, dtype=np.int64)
    e_labels = np.asarray(e_inds, dtype=np.int64)
    # convert to torch tensor
    vfeats = torch.tensor(vfeats, dtype=torch.float32)
    vfeat_lens = torch.tensor(vfeat_lens, dtype=torch.int64)
    afeats = torch.tensor(afeats, dtype=torch.float32)
    afeat_lens = torch.tensor(afeat_lens, dtype=torch.int64)
    word_ids = torch.tensor(word_ids, dtype=torch.int64)
    char_ids = torch.tensor(char_ids, dtype=torch.int64)
    s_labels = torch.tensor(s_labels, dtype=torch.int64)
    e_labels = torch.tensor(e_labels, dtype=torch.int64)
    return records, vfeats, vfeat_lens, afeats, afeat_lens, word_ids, char_ids, s_labels, e_labels


def test_collate_fn(data):
    records, video_features, audio_features, word_ids, char_ids, s_inds, e_inds, *_ = zip(*data)
    # process word ids
    word_ids, _ = pad_seq(word_ids)
    word_ids = np.asarray(word_ids, dtype=np.int32)
    # process char ids
    char_ids, _ = pad_char_seq(char_ids)
    char_ids = np.asarray(char_ids, dtype=np.int32)
    # process video features
    vfeats, vfeat_lens = pad_video_seq(video_features)
    vfeats = np.asarray(vfeats, dtype=np.float32)
    vfeat_lens = np.asarray(vfeat_lens, dtype=np.int32)
    # process audio features
    afeats, afeat_lens = pad_video_seq(audio_features)
    afeats = np.asarray(afeats, dtype=np.float32)
    afeat_lens = np.asarray(afeat_lens, dtype=np.int32)
    # convert to torch tensor
    vfeats = torch.tensor(vfeats, dtype=torch.float32)
    vfeat_lens = torch.tensor(vfeat_lens, dtype=torch.int64)
    afeats = torch.tensor(afeats, dtype=torch.float32)
    afeat_lens = torch.tensor(afeat_lens, dtype=torch.int64)
    word_ids = torch.tensor(word_ids, dtype=torch.int64)
    char_ids = torch.tensor(char_ids, dtype=torch.int64)
    s_labels = np.asarray(s_inds, dtype=np.int64)
    e_labels = np.asarray(e_inds, dtype=np.int64)
    s_labels = torch.tensor(s_labels, dtype=torch.int64)
    e_labels = torch.tensor(e_labels, dtype=torch.int64)
    return records, vfeats, vfeat_lens, afeats, afeat_lens, word_ids, char_ids, s_labels, e_labels

def get_train_loader(dataset, video_features, audio_features, configs):
    train_set = Dataset(dataset=dataset, video_features=video_features, audio_features=audio_features, mode='train')
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=configs.batch_size, shuffle=True,
                                               collate_fn=train_collate_fn, num_workers=0)
    return train_loader

def get_test_loader(dataset, video_features, audio_features, configs):
    test_set = Dataset(dataset=dataset, video_features=video_features, audio_features=audio_features)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=configs.batch_size, shuffle=False,
                                              collate_fn=test_collate_fn, num_workers=0)
    return test_loader