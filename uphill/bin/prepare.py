from uphill.bin.parser_base import (
    set_base_parser,
    add_arg_group
)
from uphill import apply
from uphill import loggerx


def set_prepare_parser(parser=None):
    if not parser:
        parser = set_base_parser()

    gp = add_arg_group(parser, 'trainer arguments')
    
    gp.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='dataset name to download')
    
    gp.add_argument(
        '--corpus_dir', 
        type=str,
        required=True, 
        help='source dataset dir'
    )
    
    gp.add_argument(
        '--target_dir', 
        type=str,
        required=True, 
        help='dir to save all documents'
    )
    
    gp.add_argument(
        '--num_jobs', 
        type=int,
        help='number of jobs to process data'
    )
    
    gp.add_argument(
        '--compress',
        action='store_true',
        default=False,
        help='compress data to gzip file',
    )
    
    return parser


def run(args, unused_args):
    loggerx.initialize()
    dataset = args.dataset
    kwargs = {
        "corpus_dir": args.corpus_dir,
        "target_dir": args.target_dir,
        "compress": args.compress
    }
    if args.num_jobs:
        kwargs["num_jobs"] = args.num_jobs
    if hasattr(apply, dataset) and hasattr(getattr(apply, dataset), "prepare"):
        getattr(apply, dataset).prepare(**kwargs)
    else:
        loggerx.warning(f"prepare operation is not valid for dataset {dataset}")

