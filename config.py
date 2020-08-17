def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default="/home/XYZ", help="make sure path refers to a somewhere on your machine")
parser.add_argument('--experiment_name', type=str, default="exp1",
                    help='An arbitarty directory name where results will be saved in PATH')
parser.add_argument('--steps', type=int, default=200,
                    help="The number of steps for running the model")
parser.add_argument('--use_coda', type=str2bool, default=True, help="Using code or not")
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.000)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--click_lstm_hidden_size', type=int, default=10)
parser.add_argument("--num_hidden_features_graph", type=int, default=10)
parser.add_argument("--num_hidden_layers_graph", type=int, default=1)
parser.add_argument('--click_lstm_num_layers', type=int, default=1)
parser.add_argument('--weeks_test', type=int, default=20, help="The number of weeks included in test")
parser.add_argument('--training_periods', type=str, default='2013J')
parser.add_argument('--testing_periods', type=str, default='2014J')
parser.add_argument('--training_courses', type=str, default='SS1')
parser.add_argument('--testing_courses', type=str, default='SS1')
parser.add_argument('--num_classes', type=int, default=2, help="The number of classes (either 2 or 4)")

args = parser.parse_args()
