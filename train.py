import os
import numpy as np
import torch
import dgl.init
from dgl import DGLGraph
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score,confusion_matrix, classification_report
from model import RGCN

import config
args =config.args
save_dir = args.path+args.experiment_name+'/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
args_dict = {}
for arg in vars(args):
    args_dict[arg] = getattr(args, arg)
with open(save_dir+'config.txt','w') as fp:
    for key in args_dict:
        fp.write("{}:{}\n".format(key,args_dict[key]))

#Course names are different in the dataset. We encoded them to reflect their type (i.e., Social Science or STEM)
course_map={'SS1':'AAA', 'SS2':'BBB', 'SS3':'GGG', 'ST1':'DDD', 'ST2':'CCC', 'ST3':'EEE', 'ST4':'FFF'}

print("Training courses {} Training periods {}".format(args.training_courses,args.training_periods))
print("Test courses {} Test periods {}".format(args.testing_courses,args.testing_periods))
def load_data():
    training_courses = [course_map[course] for course in args.training_courses.split(',')]
    training_periods = [year for year in args.training_periods.split(',')]
    test_courses = [course_map[course] for course in args.testing_courses.split(',')]
    test_periods = [year for year in args.testing_periods.split(',')]

    data_file = 'data/Data.csv'
    course_nodes = []
    student_nodes = []
    with open(data_file) as fp:
        for i, line in enumerate(fp):
            info = line.split(',')
            period = info[0]
            course = info[1]
            if (period in training_periods and course  in training_courses):
                pass
            elif (period in test_periods and course in test_courses):
                pass
            else:
                continue
            student_id = info[2]
            course_id = course
            if course_id not in course_nodes:
                course_nodes.append(course_id)
            if student_id not in student_nodes:
                student_nodes.append(student_id)
    course_nodes =list(set(course_nodes))
    student_nodes = list(set(student_nodes))
    nodes = student_nodes+course_nodes

    train_edges = []
    test_edges = []
    node_features = [ [] for _ in range(len(nodes))]
    with open(data_file) as fp:
        for i, line in enumerate(fp):
            info = line.split(',')
            period = info[0]
            course = info[1]
            if period  in  training_periods and course  in training_courses:
                pass
            elif period   in test_periods and course  in test_courses:
                pass
            else:
                continue
            course_feat = np.zeros(36)
            if course == 'AAA' or course=='BBB' or course=='GGG':
                course_feat[0] = 1
            else:
                course_feat[1] = 1
            student_id = info[2]
            course_id = course#+'_'+ period

            click_data =np.array(list(map(np.int32, info[3:803])))
            demo_data = np.array(list(map(np.int32, info[803:839])))

            if args.num_classes==2:
                y = list(map(np.int32, info[-2]))
            else:
                y = list(map(np.int32, info[-1].strip()))

            if course in training_courses and period in training_periods:
                split = 'Train'
                edge_feature = click_data
            if course in test_courses and period in test_periods:
                split='Test'
                edge_feature = np.zeros(800)
                edge_feature[0:20*args.weeks_test]= click_data[0:20*args.weeks_test]
            if split=='Test':
                test_edges.append((nodes.index(student_id),nodes.index(course_id), y[0],edge_feature, split))
            else:
                train_edges.append((nodes.index(student_id), nodes.index(course_id), y[0], edge_feature, split))
            node_features[nodes.index(student_id)] = demo_data
            node_features[nodes.index(course_id)] = course_feat

    edges = np.array(train_edges+test_edges)
    return edges, node_features

E, N = load_data()
edges=  E
node_features =N


use_cuda = args.use_coda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def run_simulation():
    ############## Train graph construction #########################
    train_graph = DGLGraph()
    train_graph.add_nodes(len(N))
    train_edges = np.array([e for e in edges if e[4]=='Train'])

    train_click_data = torch.Tensor(np.stack(train_edges[:,3])).to(device)
    train_edge_src =torch.from_numpy(train_edges[:,0].flatten().astype(int)).to(device)
    train_edge_dst =torch.from_numpy(train_edges[:,1].flatten().astype(int)).to(device)
    train_edge_type =train_edges[:,2].flatten().astype(int)
    train_edge_norm =np.ones(len(train_edge_dst), dtype=np.float32) #/ degrees.astype(np.float32)
    train_edge_type = torch.from_numpy(train_edge_type).to(device)
    train_edge_norm = torch.from_numpy(train_edge_norm).unsqueeze(1).to(device)
    train_graph.add_edges(train_edge_src, train_edge_dst, {"norm": train_edge_norm, "type": train_edge_type})
    train_graph.add_edges(train_edge_dst, train_edge_src, {"norm": train_edge_norm, "type": train_edge_type})
    train_graph.set_n_initializer(dgl.init.zero_initializer)
    train_y =torch.LongTensor(train_edges[:,2].flatten().astype(int)).to(device)
    
    ############## Test graph construction #########################

    test_graph = DGLGraph()
    test_graph.add_nodes(len(N))
    test_edges = np.array([e for e in edges if e[4]=='Test'])
    test_click_data = torch.Tensor(np.stack(test_edges[:,3])).to(device)
    test_edge_src =torch.from_numpy(test_edges[:,0].flatten().astype(int)).to(device)
    test_edge_dst =torch.from_numpy(test_edges[:,1].flatten().astype(int)).to(device)
    test_edge_type =test_edges[:,2].flatten().astype(int)
    test_edge_norm =np.ones(len(test_edge_dst), dtype=np.float32) #/ degrees.astype(np.float32)
    test_edge_type = torch.from_numpy(test_edge_type).to(device)
    test_edge_norm = torch.from_numpy(test_edge_norm).unsqueeze(1).to(device)
    test_graph.add_edges(test_edge_src, test_edge_dst, {"norm": test_edge_norm, "type": test_edge_type})
    test_graph.add_edges(test_edge_dst, test_edge_src, {"norm": test_edge_norm, "type": test_edge_type})
    test_graph.set_n_initializer(dgl.init.zero_initializer)

    test_y =torch.LongTensor(test_edges[:,2].flatten().astype(int)).to(device)
    #################################################################

    all_node_features = torch.Tensor(node_features).to(device)

    model = RGCN(
        graph=train_graph, features=all_node_features, n_hidden_feats=args.num_hidden_features_graph,
        n_hidden_layers=args.num_hidden_layers_graph, n_classes=args.num_classes,
        activation=F.relu,dropout=args.dropout,n_rels=args.num_classes, n_bases=-1,self_loop=True,
        click_lstm_hidden_size=args.click_lstm_hidden_size,click_sequence_length=40,
        click_lstm_num_layers=args.click_lstm_num_layers).to(device)



    def test():
        model.load_state_dict(torch.load(save_dir+'model.m'))
        model.eval()

        with torch.no_grad():
            output = model(all_node_features, test_click_data, test_edge_src, test_edge_dst,truncate_sequence_lstm=args.weeks_test-1)
            predictions = torch.argmax(F.softmax(output, dim=1), dim=1).cpu().numpy()
            ground_truths= test_y.cpu().numpy()
            f1 = f1_score(y_true=ground_truths,y_pred=predictions,average='weighted')
            performance_file = open(save_dir+'test_performance_log.txt','w')
            report = classification_report(y_true=ground_truths,y_pred=predictions)
            conf_matrix = confusion_matrix(y_true=ground_truths,y_pred=predictions)
            performance_file.write("classification_report:\n{}\n".format(report))
            performance_file.write("conf_matrix:\n{}\n".format(conf_matrix))
            print("Test : {}".format(f1))
            return f1
    def train():
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.99)
        criterion = nn.CrossEntropyLoss()
        for step in tqdm(range(args.steps)):
            output = model(all_node_features, train_click_data, train_edge_src, train_edge_dst,truncate_sequence_lstm=39)
            loss = criterion(output, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step %20==0 and step :
                predictions = torch.argmax(F.softmax(output, dim=1), dim=1).cpu().numpy()
                ground_truths= train_y.cpu().numpy()
                train_f1 = f1_score(y_true=ground_truths,y_pred=predictions,average='weighted')
                print('Step {} Loss {} Train F1-score {}'.
                      format(step, loss.data, train_f1))
                model.train()
        torch.save(model.state_dict(), save_dir + 'model.m')
    train()
    test()

run_simulation()

