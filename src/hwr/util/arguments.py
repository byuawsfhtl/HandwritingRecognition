def parse_inference_arguments(args):
    """
    Function to parse command line arguments for inference

    :param args: command line arguments passed to the inference python file
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
        elif args[index] == '--weights_path':
            index += 1
            arg_dict['weights_path'] = args[index]
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
    if 'weights_path' not in arg_dict:
        raise Exception('The weights_path argument must be specified')

    # Set arguments to their defaults if not present on command line
    if 'console' not in arg_dict:
        arg_dict['console'] = False
    if 'log_level' not in arg_dict:
        arg_dict['log_level'] = '3'

    return arg_dict


def parse_train_arguments(args):
    """
    Function to parse command line arguments for training

    :param args: command line arguments passed to the train python file
    :return: dictionary with configuration settings
    """
    arg_dict = dict()

    index = 0
    while index < len(args):
        if args[index] == '--img_path':
            index += 1
            arg_dict['img_path'] = args[index]
        elif args[index] == '--label_path':
            index += 1
            arg_dict['label_path'] = args[index]
        elif args[index] == '--show_graphs':
            arg_dict['show_graphs'] = True
        elif args[index] == '--log_level':
            index += 1
            arg_dict['log_level'] = args[index]
        elif args[index] == '--model_out':
            index += 1
            arg_dict['model_out'] = args[index]
        elif args[index] == '--epochs':
            index += 1
            arg_dict['epochs'] = int(args[index])
        elif args[index] == '--batch_size':
            index += 1
            arg_dict['batch_size'] = int(args[index])
        elif args[index] == '--learning_rate':
            index += 1
            arg_dict['learning_rate'] = float(args[index])
        elif args[index] == '--max_seq_size':
            index += 1
            arg_dict['max_seq_size'] = int(args[index])
        elif args[index] == '--train_size':
            index += 1
            arg_dict['train_size'] = float(args[index])
        elif args[index] == '--tfrecord_out':
            index += 1
            arg_dict['tfrecord_out'] = args[index]
        elif args[index] == '--weights_path':
            index += 1
            arg_dict['weights_path'] = args[index]
        else:
            raise Exception('Unexpected command line argument:' + args[index])

        index += 1

    # Required command line arguments
    if 'img_path' not in arg_dict:
        raise Exception('The img_path argument must be set!')
    if 'label_path' not in arg_dict:
        raise Exception('The label_path argument must be set!')

    # Set arguments to their defaults if not present on command line
    if 'show_graphs' not in arg_dict:
        arg_dict['show_graphs'] = False
    if 'log_level' not in arg_dict:
        arg_dict['log_level'] = '3'
    if 'model_out' not in arg_dict:
        arg_dict['model_out'] = './data/model_weights/hwr_model/run1'
    if 'epochs' not in arg_dict:
        arg_dict['epochs'] = 100
    if 'batch_size' not in arg_dict:
        arg_dict['batch_size'] = 100
    if 'learning_rate' not in arg_dict:
        arg_dict['learning_rate'] = 4e-4
    if 'max_seq_size' not in arg_dict:
        arg_dict['max_seq_size'] = 128
    if 'train_size' not in arg_dict:
        arg_dict['train_size'] = .8
    if 'tfrecord_out' not in arg_dict:
        arg_dict['tfrecord_out'] = './data/misc/data.tfrecords'
    if 'weights_path' not in arg_dict:
        arg_dict['weights_path'] = None

    return arg_dict
