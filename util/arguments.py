def parse_arguments(args):
    """
    Function to parse command line arguments

    :param args: The command line arguments passed to the entry point python file
    :return: dictionary with configuration settings
    """
    arg_dict = dict()

    index = 0
    while index < len(args):
        if args[index] == '--img_path':
            index += 1
            arg_dict['img_path'] = args[index]
        elif args[index] == '--out_path':
            index += 1
            arg_dict['out_path'] = args[index]
        elif args[index] == '--log_level':
            index += 1
            assert args[index] in ['0', '1', '2', '3']  # log_level must be 0-3
            arg_dict['log_level'] = args[index]
        elif args[index] == '--console':
            arg_dict['console'] = True
        else:
            raise Exception('Unexpected command line argument:' + args[index])

        index += 1

    # Required command line arguments
    if 'img_path' not in arg_dict:
        raise Exception('The img_path argument must be specified!')
    if 'out_path' not in arg_dict and 'console' not in arg_dict:
        raise Exception('The out_path argument must be specified if console output is not specified')

    # Set arguments to their defaults if not present on command line
    if 'console' not in arg_dict:
        arg_dict['console'] = False
    if 'log_level' not in arg_dict:
        arg_dict['log_level'] = '3'

    return arg_dict
