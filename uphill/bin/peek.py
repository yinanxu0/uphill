from uphill.bin.parser_base import (
    set_base_parser,
    add_arg_group
)
from uphill import loggerx
from uphill import DocumentArray, SupervisionArray

def set_peek_parser(parser=None):
    if not parser:
        parser = set_base_parser()

    gp = add_arg_group(parser, 'peek arguments')
    
    gp.add_argument(
        '-i',
        '--path',
        type=str,
        required=True,
        help='document arry path to be peeked')
    
    gp.add_argument(
        '-n',
        '--num', 
        type=int,
        default=1,
        help='peeked number'
    )
    
    return parser


def run(args, unused_args):
    loggerx.initialize()
    num = max(args.num, 1)
    candidates = [DocumentArray, SupervisionArray]
    data_set = None
    for manifest in candidates:
        try:
            data_set = manifest.from_file(args.path)
            if len(data_set) == 0:
                raise RuntimeError()
            break
        except Exception:
            pass
    if data_set is None:
        raise ValueError(f"Unknown type of manifest: {args.path}")
    for i in range(num):
        loggerx.info(f"Document {i} infomation...")
        data_set[i].summary()

