import argparse
from termcolor import colored


from uphill import __version__


def add_arg_group(parser, title):
    return parser.add_argument_group(title)


def set_base_parser():
    parser = argparse.ArgumentParser(
        epilog='%s, a toolkit to process dataset easily. '
        'Visit %s for tutorials and documents.' % (
            colored('uphill v%s' % __version__, 'green'),
            colored(
                'https://github.com/yinanxu0/uphill',
                'cyan',
                attrs=['underline'],
            ),
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Uphill Line Interface',
    )

    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=__version__,
        help='show UpHill version',
    )

    return parser

