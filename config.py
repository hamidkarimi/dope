def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default="/home/XYZ", help="make sure path refers to somewhere on your machine")
parser.add_argument('--experiment_name', type=str, default="exp1",
                    help='An arbitarty name for a directory  where results will be saved (PATH/experiment_name)')
parser.add_argument('--steps', type=int, default=200,
                    help="The number of steps for running the model")
parser.add_argument('--use_coda', type=str2bool, default=True, help="Using coda (gpu) or not")
parser.add_argument('--learning_rate', type=float, default=0.01, help="learning rate of optimization")
parser.add_argument('--weight_decay', type=float, default=0.000, help="L2 penalty coefficient")
parser.add_argument('--dropout', type=float, default=0.4, help="Dropout")
parser.add_argument('--click_lstm_hidden_size', type=int, default=10, help="Number of hidden neurons in LSTM network modeling the click data")
parser.add_argument('--click_lstm_num_layers', type=int, default=1, help="Number of LSTM network modeling the click data")
parser.add_argument("--num_hidden_features_graph", type=int, default=10, help="Number of hidden neurons in RGCN")
parser.add_argument("--num_hidden_layers_graph", type=int, default=1, help="Number of aggregation layers in RGCN")
parser.add_argument('--weeks_test', type=int, default=20, help="The number of weeks included in test")
parser.add_argument('--training_periods', type=str, default='2013J', help="Training periods (semesters/terms) seperate them by comma")
parser.add_argument('--testing_periods', type=str, default='2014J', help="Testing periods (semesters/terms) seperate them by comma")
parser.add_argument('--training_courses', type=str, default='SS1', help="Courses to train the model on seperate them by comma ")
parser.add_argument('--testing_courses', type=str, default='SS1', help="Courses to test the model on seperate them by comma ")
parser.add_argument('--num_classes', type=int, default=2, help="The number of classes. either 2 (pass/fail)  or 4 (pass/fail/disctinction/withdrawn)")

args = parser.parse_args()
