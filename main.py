import numpy as np
import json
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import importlib
import logging
import os
import copy
import datetime
import random

from data_mimic import MimicFullDataset, my_collate_fn, my_collate_fn_led, DataCollatorForMimic, modify_rule
from torch.nn.parallel import DistributedDataParallel

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from networks import LongformerForMaskedLM 
from utils import *
from dataset_utils import partition_data, get_dataloader


def get_args():
    # general arguments for all methods
    parser = argparse.ArgumentParser()

    # federated setup parameters
    parser.add_argument('--model', type=str, default='LongformerForMaskedLM',
                        help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='MIMIC3',
                        help='dataset used for training')
    parser.add_argument('--partition', type=str, default='homo',
                        help='the data partitioning strategy')
    # parser.add_argument('--alpha', type=float, default=0.5,
    #                     help='concentration parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--n_parties', type=int, default=10,
                        help='number of workers in a distributed cluster')
    parser.add_argument('--sample_fraction', type=float, default=1.0,
                        help='how many clients are sampled in each round')
    parser.add_argument('--approach', type=str, default='fedavg',
                        help='federated learning algorithm being used')
    parser.add_argument('--n_comm_round', type=int, default=50,
                        help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0,
                        help="Random seed")

    # local training parameters
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--rho', type=float, default=0.9,
                        help='momentum of sgd optimizer')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of local epochs')   
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help="weight decay during local training")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='the optimizer')
    parser.add_argument('--auto_aug', action='store_true',
                        help='whether to apply auto augmentation')

    # logging parameters
    parser.add_argument('--print_interval', type=int, default=50,
                        help='how many comm round to print results on screen')
    parser.add_argument('--datadir', type=str, required=False, default="./data/",
                        help="Data directory")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/",
                        help='Log directory path')
    parser.add_argument('--log_file_name', type=str, default=None,
                        help='The log file name')

    parser.add_argument('--ckptdir', type=str, required=False, default="./models/",
                        help='directory to save model')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='how many rounds do we save the checkpoint one time')

    args, appr_args = parser.parse_known_args()
    return args, appr_args


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    version: Optional[str] = field(
        default=None, metadata={"help": "mimic version"}
    )
    global_attention_strides: Optional[int] = field(
        default=3,
        metadata={
            "help": "how many gap between each (longformer) golabl attention token in prompt code descriptions, set to 1 for maximum accuracy, but requires more gpu memory."
        },
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    finetune_terms: str = field(
        default="no",
        metadata={"help": "what terms to train like bitfit (bias)."},
    )

def init_nets(n_parties, args):
    nets = {net_i: None for net_i in range(n_parties)}
    if args.dataset in {'mnist', 'cifar10', 'svhn'}:
        n_classes = 10
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    else:
        raise NotImplementedError('dataset not supported')

    for net_i in range(n_parties):
        if args.model == 'resnet20':
            net = resnet20(num_classes=n_classes)
        elif args.model == 'resnet32':
            net = resnet32(num_classes=n_classes)
        elif args.model == 'resnet18':
            net = resnet18(num_classes=n_classes)
        elif args.model == 'mobilenetv2':
            net = mobilenetv2(num_classes=n_classes)
        else:
            raise NotImplementedError('model not supported')
        nets[net_i] = net

    return nets

def parse_args_kept():
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.broadcast_buffers = False
        
    # Setup logging
    # if is_main_process(training_args.local_rank):
    #     wandb.init(project="mimic_coder", entity="whaleloops")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if training_args.should_log else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    return model_args, data_args, training_args

def load_checkpoint_kept(training_args):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


if __name__ == '__main__':
    # ===== parsing arguments and initialize method =====
    model_args, data_args, training_args = parse_args_kept()

    last_checkpoint = load_checkpoint_kept(training_args)

    




    if args.approach == 'fedavg':
        from approach.fedavg import FedAvg as Appr
    elif args.approach == 'fedprox':
        from approach.fedprox import FedProx as Appr
    elif args.approach == 'fedsam':
        from approach.fedsam import FedSAM as Appr
    elif args.approach == 'fedlogitcal':
        from approach.fedlogitcal import FedLogitCal as Appr
    elif args.approach == 'fedrs':
        from approach.fedrs import FedRS as Appr
    elif args.approach == 'fedoptim':
        from approach.fedoptim import FedOptim as Appr
    elif args.approach == 'fednova':
        from approach.fednova import FedNova as Appr
    elif args.approach == 'moon':
        from approach.moon import MOON as Appr
    elif args.approach == 'fedexp':
        from approach.fedexp import FedExp as Appr
    else:
        raise NotImplementedError('approach not implemented')

    # arguments specific to the chosen FL algorithm
    appr_args = Appr.extra_parser(appr_args)
    # ===================================================


    # ================ logging related ==================
    mkdirs(args.logdir)
    mkdirs(args.ckptdir)
    mkdirs(os.path.join(args.ckptdir, args.approach))

    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    argument_path = argument_path + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args) + str(appr_args), f)
    print(str(args))
    print(str(appr_args))

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    print('log path: ', log_path)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    # ===================================================


    # ================ dataset related ==================
    logger.info("Partitioning data")
    seed_everything(args.init_seed)
    # mapping from individual client to sample idx of the whole dataset
    party2dataidx = partition_data(
        args.dataset, args.datadir, args.partition, args.n_parties, alpha=args.alpha)
    # mapping from individual client to its local training data loader
    party2loaders = {}
    for party_id in range(args.n_parties):
        train_dl_local, _ = get_dataloader(args, args.dataset, args.datadir,
            args.batch_size, args.batch_size, party2dataidx[party_id])
        party2loaders[party_id] = train_dl_local
    # these loaders are used for evaluating accuracy of global model
    global_train_dl, test_dl = get_dataloader(args, args.dataset, args.datadir,
                            train_bs=args.batch_size, test_bs=args.batch_size)

    # support random party sampling
    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []
    if n_party_per_round != args.n_parties:
        for i in range(args.n_comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.n_comm_round):
            party_list_rounds.append(party_list)
    # ===================================================


    # ================ network related ================
    logger.info("Initializing nets")
    party2nets = init_nets(args.n_parties, args)
    global_net = init_nets(1, args)[0]
    # =================================================

    # ================ run FL ================
    fed_alg = Appr(args, appr_args, logger, party_list_rounds,
                party2nets, global_net,
                party2loaders, global_train_dl, test_dl)
    fed_alg.run_fed()
    # ========================================
