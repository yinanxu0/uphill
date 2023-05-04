import sys

__all__ = ['main']


def _get_run_args(print_args: bool = True):

    from .parser import get_main_parser

    parser = get_main_parser()
    if len(sys.argv) > 1:
        args, unused_args = parser.parse_known_args()

        return args, unused_args
    else:
        parser.print_help()
        exit()


def main():
    """The main entrypoint of the CLI."""
    from uphill import bin
    args, unused_args = _get_run_args()
    getattr(bin, args.cli.replace('-', '_')).run(args, unused_args)



if __name__ == '__main__':
    main()
