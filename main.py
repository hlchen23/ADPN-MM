import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from model.ADPNet import ADPNet, build_optimizer_and_scheduler
from util.data_util import load_video_features, save_json, load_json, load_audio_features
from util.data_gen import gen_or_load_dataset
from util.data_loader import get_train_loader, get_test_loader
from util.runner_utils import set_th_config, convert_length_to_mask, eval_test, filter_checkpoints, \
    get_last_checkpoint, grader
import numpy as np

import faulthandler
faulthandler.enable()

parser = argparse.ArgumentParser()

parser.add_argument('--ext', type=int, default=3, help='Expansion coefficient for Curriculum Optimization Strategy')
parser.add_argument('--rho_th', type=float, default=0.3, help='threshold for rho')
parser.add_argument('--s_v_th', type=float, default=0.25, help='threshold for s_v')
parser.add_argument('--lamb', type=float, default=25, help='the lambda in loss function')
parser.add_argument('--test_path', type=str, default=None, help='the path of the trained model when testing')
parser.add_argument("--save_predictions",type=str,default=None,help='the path to save predictions')
parser.add_argument("--nse_num",type=int,default=3,help="number of semantics entity")

# data parameters
parser.add_argument('--save_dir', type=str, default='datasets', help='path to save processed dataset')
parser.add_argument('--task', type=str, default='charades', help='target task, [charades|activitynet]')
# predict clips are cut by visual
parser.add_argument('--max_pos_len', type=int, default=128, help='maximal position sequence length allowed for Visual, 128 for c3d, 128*3 for VGG')
parser.add_argument('--max_pos_len_a', type=int, default=128*3, help='maximal position sequence length allowed for Audio, 128 for VGGish, 128*3 for PANN')
# model parameters
parser.add_argument("--word_size", type=int, default=None, help="number of words")
parser.add_argument("--char_size", type=int, default=None, help="number of characters")
parser.add_argument("--word_dim", type=int, default=300, help="word embedding dimension")
parser.add_argument("--video_feature_dim", type=int, default=1024, help="video feature input dimension, 1024 for c3d, 4096 for VGG")
parser.add_argument("--audio_feature_dim", type=int, default=128, help="audio feature input dimension, 128 for VGGish, 2048 for PANN")
parser.add_argument("--char_dim", type=int, default=50, help="character dimension, set to 100 for activitynet")
parser.add_argument("--dim", type=int, default=128, help="hidden size")
parser.add_argument("--num_heads", type=int, default=8, help="number of heads")
parser.add_argument("--drop_rate", type=float, default=0.2, help="dropout rate")
# training/evaluation parameters
parser.add_argument("--gpu_idx", type=str, default="0", help="GPU index")
parser.add_argument("--seed", type=int, default=12345, help="random seed")
parser.add_argument("--mode", type=str, default="train", help="[train | test]")
parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
parser.add_argument("--num_train_steps", type=int, default=None, help="number of training steps")
parser.add_argument("--init_lr", type=float, default=0.0001, help="initial learning rate")
parser.add_argument("--clip_norm", type=float, default=1.0, help="gradient clip norm")
parser.add_argument("--warmup_proportion", type=float, default=0.0, help="warmup proportion")
parser.add_argument("--period", type=int, default=100, help="training loss print period")
parser.add_argument('--model_dir', type=str, default='ckpt', help='path to save trained model weights')
parser.add_argument('--model_name', type=str, default='ADPNet', help='model name')
parser.add_argument('--suffix', type=str, default=None, help='set to the last `_xxx` in ckpt repo to eval results')
configs = parser.parse_args()

if configs.task == 'charades':
    # I3D visual features
    configs.max_pos_len = int(128 * 1.5)
    configs.video_feature_dim = 1024
    visual_features = load_video_features(os.path.join('data', 'features', configs.task, "i3d_video"), configs.max_pos_len)
    # PANNs audio features
    configs.max_pos_len_a = 128 * 3
    configs.audio_feature_dim = 2048
    audio_features = load_audio_features(os.path.join('data', 'features', configs.task, "audio"), configs.max_pos_len_a, mode='PANN')
else:
    # C3D visual features
    configs.max_pos_len = 128
    configs.video_feature_dim = 1024
    visual_features = load_video_features(os.path.join('data', 'features', configs.task, 'c3d_video'), configs.max_pos_len)
    # VGGish audio features
    configs.max_pos_len_a = int(128 / 1.8)
    configs.audio_feature_dim = 128
    audio_features = load_audio_features(os.path.join('data', 'features', configs.task, "audio/VGGish.pickle"), configs.max_pos_len_a, mode='VGGish')

set_th_config(configs.seed)
dataset = gen_or_load_dataset(configs)
configs.char_size = dataset['n_chars']
configs.word_size = dataset['n_words']
train_loader = get_train_loader(dataset=dataset['train_set'], video_features=visual_features, audio_features=audio_features, configs=configs)
val_loader = None if dataset['val_set'] is None else get_test_loader(dataset=dataset['val_set'], video_features=visual_features, audio_features=audio_features, configs=configs)
test_loader = get_test_loader(dataset=dataset['test_set'], video_features=visual_features, audio_features=audio_features,configs=configs)
configs.num_train_steps = len(train_loader) * configs.epochs
num_train_batches = len(train_loader)
num_val_batches = 0 if val_loader is None else len(val_loader)
num_test_batches = len(test_loader)

# Device configuration
cuda_str = 'cuda' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx)
device = torch.device(cuda_str if torch.cuda.is_available() else 'cpu')

# create model dir
home_dir = os.path.join(configs.model_dir, '_'.join([configs.model_name, configs.task,
                                                     'v_'+str(configs.max_pos_len), 'a_'+str(configs.max_pos_len_a)]))
if configs.suffix is not None:
    home_dir = home_dir + '_' + configs.suffix
model_dir = os.path.join(home_dir, "model")

# train and test
if configs.mode.lower() == 'train':
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    eval_period = num_train_batches // 2
    save_json(vars(configs), os.path.join(model_dir, 'configs.json'), sort_keys=True, save_pretty=True)
    # build model
    model = ADPNet(configs=configs, word_vectors=dataset['word_vector']).to(device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, configs=configs) # warm_up and weight_reduce are related to num_train_steps
    # start training
    best_r1i7 = -1.0
    score_writer = open(os.path.join(model_dir, "eval_results.txt"), mode="w", encoding="utf-8")
    print('start training...', flush=True)
    global_step = 0
    for epoch in range(configs.epochs):
        model.train()
        for data in tqdm(train_loader, total=num_train_batches, desc='Epoch %3d / %3d' % (epoch + 1, configs.epochs)):
            # len(train_loader) batches
            global_step += 1
            _ , vfeats, vfeat_lens, afeats, afeat_lens, word_ids, char_ids, s_labels, e_labels = data
            # prepare features
            vfeats, vfeat_lens = vfeats.to(device), vfeat_lens.to(device)
            afeats, afeat_lens = afeats.to(device), afeat_lens.to(device)
            word_ids, char_ids = word_ids.to(device), char_ids.to(device)
            s_labels, e_labels = s_labels.to(device), e_labels.to(device)
            # generate mask
            query_mask = (torch.zeros_like(word_ids) != word_ids).float().to(device)
            video_mask = convert_length_to_mask(vfeat_lens).to(device)
            audio_mask = convert_length_to_mask(afeat_lens).to(device)
            # compute logits
            start_logits_av, end_logits_av, start_logits_v, end_logits_v, se_attw = \
                model(word_ids, char_ids, vfeats, video_mask, afeats, audio_mask, query_mask)
            
            rho, s_av, s_v = grader(
                start_logits_av, end_logits_av,
                start_logits_v, end_logits_v,
                s_labels, e_labels, ext = configs.ext
            )

            loc_loss = model.compute_weighted_loss(
                start_logits_av, end_logits_av,
                start_logits_v, end_logits_v,
                s_labels, e_labels,
                grades = (rho, s_av, s_v),
                rho_th=configs.rho_th,
                s_v_th=configs.s_v_th
            )
        
            loc_loss_av = model.compute_loss(start_logits_av, end_logits_av, s_labels, e_labels)
            loc_loss_v = model.compute_loss(start_logits_v, end_logits_v, s_labels, e_labels)
            dqa_loss = model.compute_dqa_loss(se_attw)
            
            bs = rho.shape[0]
            total_loss = loc_loss + configs.lamb * dqa_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), configs.clip_norm)
            optimizer.step()
            scheduler.step()
            
            if global_step % eval_period == 0 or global_step % num_train_batches == 0:
                model.eval()
                r1i3, r1i5, r1i7, mi, score_str = eval_test(model=model, data_loader=test_loader, device=device,
                                                            mode='test', epoch=epoch + 1, global_step=global_step,configs=configs)
                print('\nEpoch: %2d | Step: %5d | r1i3: %.2f | r1i5: %.2f | r1i7: %.2f | mIoU: %.2f' % (
                    epoch + 1, global_step, r1i3, r1i5, r1i7, mi), flush=True)
                score_writer.write(score_str)
                score_writer.flush()
                if r1i7 >= best_r1i7:
                    best_r1i7 = r1i7
                    torch.save(model.state_dict(), os.path.join(model_dir, '{}_{}.t7'.format(configs.model_name,
                                                                                            global_step)))
                    filter_checkpoints(model_dir, suffix='t7', max_to_keep=3)
                model.train()          

    score_writer.close()

elif configs.mode.lower() == 'test':
    if configs.test_path != None:
        model_dir = configs.test_path
    if not os.path.exists(model_dir):
        raise ValueError('No pre-trained weights exist')
    pre_configs = load_json(os.path.join(model_dir, "configs.json"))
    parser.set_defaults(**pre_configs)
    configs = parser.parse_args()
    model = ADPNet(configs=configs, word_vectors=dataset['word_vector']).to(device)
    filename = model_dir + '/' + configs.task + '_ckpt' + '.t7' # charades_ckpt.t7 / activitynet_ckpt.t7
    model.load_state_dict(torch.load(filename,map_location=device),strict=False)
    model.eval()
    r1i3, r1i5, r1i7, mi, _ = eval_test(model=model, data_loader=test_loader, device=device, mode='test',configs=configs)
    print("\n" + "\x1b[1;31m" + "Rank@1, IoU=0.3:\t{:.2f}".format(r1i3) + "\x1b[0m", flush=True)
    print("\x1b[1;31m" + "Rank@1, IoU=0.5:\t{:.2f}".format(r1i5) + "\x1b[0m", flush=True)
    print("\x1b[1;31m" + "Rank@1, IoU=0.7:\t{:.2f}".format(r1i7) + "\x1b[0m", flush=True)
    print("\x1b[1;31m" + "{}:\t{:.2f}".format("mean IoU".ljust(15), mi) + "\x1b[0m", flush=True)
