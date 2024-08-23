from uphill.bin.parser_base import set_base_parser
from uphill.bin.download import set_download_parser
from uphill.bin.prepare import set_prepare_parser
from uphill.bin.peek import set_peek_parser


def get_main_parser():
    # create the top-level parser
    parser = set_base_parser()

    sp = parser.add_subparsers(
        dest='cli',
        description='use "%(prog)-8s [sub-command] --help" '
        'to get detailed information about each sub-command',
    )

    set_download_parser(
        sp.add_parser(
            'download',
            help='⏬ download a dataset automatically',
            description='Easily to download a dataset, '
            'without any extra codes.',
        )
    )
    
    set_prepare_parser(
        sp.add_parser(
            'prepare',
            help='👋 prepare a dataset automatically',
            description='Easily to prepare a dataset, '
            'without any extra codes.',
        )
    )

    set_peek_parser(
        sp.add_parser(
            'peek',
            help='👀 peek a dataset quickly',
            description='Easily to peek a dataset'
        )
    )

    return parser
