import argparse
import random
import datetime
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from transformers import BertConfig, BertGenerationEncoder, RobertaModel, PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2Tokenizer, EncoderDecoderModel, EncoderDecoderConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments, get_cosine_schedule_with_warmup
import torch.optim as optim
import numpy as np
import pickle
from data_loader import *
from model import Model
import os
import time
from tqdm import tqdm
from rouge_score import rouge_scorer


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('-dn', '--dataset_name', type=str, default='pubchemkg')

    parser.add_argument('-ne', '--num_epochs', type=int, default=100)
    parser.add_argument('-ls', '--log_steps', type=int, default=10)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-flr', '--finetune_lr', type=float, default=1e-5)
    parser.add_argument('-ms', '--minibatch_size', type=int, default=64)
    parser.add_argument('-tr', '--training_ratio', type=float, default=0.8)
    parser.add_argument('-mml', '--max_mol_length', type=int, default=128)
    parser.add_argument('-mtl', '--max_text_length', type=int, default=128)
    parser.add_argument('-nl', '--num_conv_layers', type=int, default=2)
    parser.add_argument('-nn', '--num_sampled_neighbors', type=int, default=5)
    parser.add_argument('-neg', '--num_negative_samples', type=int, default=5)
    parser.add_argument('-reg_kge', '--reg_kge', type=float, default=1)
    parser.add_argument('-agg', '--aggregator', type=str, default='vt')

    parser.add_argument('-ddp', '--distributed_training', type=bool, default=True)
    parser.add_argument('-gpu', '--gpu', type=int, default=0, help='used only when ddp is False')
    parser.add_argument('-rs', '--random_seed', type=int, default=519)

    return parser.parse_args()


def set_random_seed(random_seed):

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.random.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def cleanup():

    dist.destroy_process_group()


def reduce_tensor(tensor: torch.Tensor):

    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size()

    return rt


def get_optimizer_and_schedule(args, model):

    init_params_id = []
    for layer in model.encoder.encoder.graph_conv_layer:
        for layer_2 in layer.conv_layers:
            init_params_id.extend(list(map(id, layer_2.parameters())))
        #init_params_id.extend(list(map(id, layer.conv_layers.parameters())))
    init_params_id.extend(list(map(id, model.encoder.encoder.entity_emb.parameters())))
    init_params_id.extend(list(map(id, model.encoder.encoder.relation_emb.parameters())))
    init_params_id.extend(list(map(id, model.encoder.encoder.relation_matrix)))
    pretrained_params = filter(
        lambda p: id(p) not in init_params_id, model.parameters()
    )
    initialized_params = filter(lambda p: id(p) in init_params_id, model.parameters())
    params_setting = [
        {"params": initialized_params},
        {"params": pretrained_params, "lr": args.finetune_lr},
    ]
    optimizer = optim.Adam(params_setting, lr=args.learning_rate)
    schedule = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=args.num_epochs,
        num_warmup_steps=args.num_epochs,
    )
    return optimizer, schedule


def load_data(args):

    data_center = DataCenter(args)
    training_data, test_data = Data(args, 'train', data_center), Data(args, 'test', data_center)
    training_sampler = DistributedSampler(training_data, shuffle=True) if args.distributed_training else RandomSampler(training_data)
    test_sampler = SequentialSampler(test_data)
    training_loader = DataLoader(training_data, batch_size=args.minibatch_size, sampler=training_sampler)
    test_loader = DataLoader(test_data, batch_size=args.minibatch_size, sampler=test_sampler)

    return data_center, training_loader, test_loader


def train(args):

    # Setup CUDA, GPU & distributed training
    if not args.distributed_training:
        args.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        args.local_rank = -1
        args.ngpu = 1
    else:
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=36000))
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.global_rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.ngpu = args.world_size
        torch.cuda.set_device(args.local_rank)
        torch.cuda.empty_cache()
        args.device = torch.device('cuda:' + str(args.local_rank))

    set_random_seed(args.random_seed + args.local_rank)

    if args.local_rank in [-1, 0]:
        print('******************************************************')
        print('********************* fine-tuning ********************')
        print('******************************************************')

    if args.local_rank in [-1, 0]:
        print('Loading data...')
    data_center, training_loader, test_loader = load_data(args)

    if args.local_rank in [-1, 0]:
        print('Loading model...')
    model = Model(args, data_center, data_center.bert_tokenizer, data_center.gpt2_tokenizer)
    model = model.to(args.device)

    # define DDP here
    if args.distributed_training:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    else:
        ddp_model = model

    if args.ngpu > 1:
        optimizer, scheduler = get_optimizer_and_schedule(args, ddp_model.module.model)
    else:
        optimizer, scheduler = get_optimizer_and_schedule(args, ddp_model.model)

    if args.local_rank in [-1, 0]:
        print('Start fine-tuning...')

    t = time.time()
    for epoch_idx in range(1, args.num_epochs + 1):
        # training
        one_epoch_loss, one_epoch_gen_loss, one_epoch_kge_loss = 0.0, 0.0, 0.0
        ddp_model.train()
        if args.distributed_training:
            training_loader.sampler.set_epoch(epoch_idx)
        data_center.sample_neighbors()
        for idx, batch in tqdm(enumerate(training_loader), total=len(training_loader)):
            input_ids, attention_mask, labels, mol_ids, _ = batch
            res = ddp_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, mol_ids=mol_ids)
            loss, gen_loss, kge_loss = res[0], res[1], res[2]
            loss.backward()
            optimizer.step()
            scheduler.step()
            ddp_model.zero_grad()
            with torch.no_grad():
                one_epoch_loss = loss.item()
                one_epoch_gen_loss = gen_loss.item()
                one_epoch_kge_loss = kge_loss.item()
            if args.distributed_training:
                torch.distributed.barrier()

        # testing
        if epoch_idx % args.log_steps == 0 and args.local_rank in [-1, 0]:
            print('******************************************************')
            print('Time: %ds' % (time.time() - t), '\tEpoch: %d/%d' % (epoch_idx, args.num_epochs), '\tLoss: %f' % one_epoch_loss)
            print(one_epoch_loss, one_epoch_gen_loss, one_epoch_kge_loss)
            ckpt_folder_exists = os.path.exists('./ckpt/' + args.dataset_name)
            if not ckpt_folder_exists:
                os.makedirs('./ckpt/' + args.dataset_name)
            agg = 'sum' if args.aggregator == 'sum' else 'vt'
            torch.save(model.state_dict(), './ckpt/' + args.dataset_name + '/pretrain_molcap_' + '_' + agg + '.pt')
            eval(args, model, test_loader, data_center.gpt2_tokenizer)

        if args.distributed_training:
            torch.distributed.barrier()


def eval(args, model, test_loader, gpt2_tokenizer):

    hyp, ref = [], []
    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader, total=len(test_loader)):
            input_ids, attention_mask, labels, mol_ids, _ = data
            outputs = model.inference(input_ids=input_ids, attention_mask=attention_mask, mol_ids=mol_ids)
            hypoth = gpt2_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            labels[labels == -100] = gpt2_tokenizer.pad_token_id
            refer = gpt2_tokenizer.batch_decode(labels, skip_special_tokens=True)
            hyp.extend(hypoth)
            ref.extend(refer)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    rouge_scores = []
    for i in range(len(hyp)):
        rs = scorer.score(hyp[i], ref[i])
        rouge_scores.append(rs)

    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores])
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores])
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores])

    print('Rouge-1: %.4f' % rouge_1)
    print('Rouge-2: %.4f' % rouge_2)
    print('Rouge-L: %.4f' % rouge_l)


def main(args):

    if args.mode == 'train':
        train(args)
    else:
        ################## You should use single GPU for testing. ####################
        print('******************************************************')
        print('********************** testing ***********************')
        print('******************************************************')
        args.distributed_training = False
        args.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        args.local_rank = -1
        set_random_seed(args.random_seed)
        data_center, training_loader, test_loader = load_data(args)
        model = Model(args, data_center, data_center.bert_tokenizer, data_center.gpt2_tokenizer)
        model = model.to(args.device)
        agg = 'sum' if args.aggregator == 'sum' else 'vt'
        ckpt = torch.load('./ckpt/' + args.dataset_name + '/pretrain_molcap_' + '_' + agg + '.pt', map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
        eval(args, model, test_loader, data_center.gpt2_tokenizer)


if __name__ == '__main__':
    main(parse_args())