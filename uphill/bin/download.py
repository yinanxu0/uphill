from uphill.bin.parser_base import (
    set_base_parser,
    add_arg_group
)
from uphill import apply
from uphill import loggerx

def set_download_parser(parser=None):
    if not parser:
        parser = set_base_parser()

    gp = add_arg_group(parser, 'trainer arguments')
    
    gp.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='dataset name to download')
    
    gp.add_argument(
        '--target_dir', 
        type=str,
        required=True, 
        help='dataset saved dir'
    )
    
    gp.add_argument(
        '--url',
        default=None,
        type=str,
        help='url to download dataset'
    )
    
    gp.add_argument(
        '--force_download',
        action='store_true',
        default=False,
        help='force to download dataset even if dataset exists',
    )
    
    return parser


def run(args, unused_args):
    loggerx.initialize()
    dataset = args.dataset
    kwargs = {
        "target_dir": args.target_dir,
        "force_download": args.force_download
    }
    if args.url is not None:
        kwargs["url"] = args.url
    if hasattr(apply, dataset) and hasattr(getattr(apply, dataset), "download"):
        getattr(apply, dataset).download(**kwargs)
    else:
        loggerx.warning(f"download operation is not valid for dataset {dataset}")
