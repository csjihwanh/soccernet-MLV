import torch
import json
import numpy as np
import torch_geometric
from torch_geometric.data import Data
import networkx as nx 
import matplotlib.pyplot as plt

from .graph_utils import generate_intra_spatial_edge, visualize_graph
#from .graph_interpolation import graph_interpolation_null

class PoseGraph():
    """
    The graph to model the sekeltons extracted by previous models
    
    Args:
        strategy

        layout

        max_)hop
        dilation 
    
    """
    def __init__(self, pose = None, interpolation_method = 'null', debug =False):

        self.debug =debug # debugging purpose

        self.pose = pose
        self.pose_dict = dict() # pose_dict[(frame_num, player_num)] = 17*3 features
        self.graph_dict = dict() # graph_dict[(frame_num, player_num)] = one pose graph\
        self.index = dict()

        self.frame_graph_list = [] # frame_graph_list[(frame_num)] = one frame graph

        #self.top_player_num = top_player_num; # hyperparameter
        self.num_joint = 14 # follows COCO pose '18
    
        #self.graph_interpolation = graph_interpolation_null()
        
    def generate_pose_dict(self):
        for frame_data in self.pose:
            frame_id = frame_data['frame_id'] 
            print("Checking frame data:", frame_data)
            for idx, instance in enumerate(frame_data['instances']):
                print("Instance type:", type(instance))
                self.pose_dict[(frame_id, idx)] = instance['keypoints']

    def generate_graph(self):

        if self.debug:
            
            num_nodes = 2125
            num_node_features = 3
            num_edges = 17330
            num_edge_features = 3

            x = torch.rand(num_nodes, num_node_features)
            edge_index = torch.randint(0, num_nodes, (2, num_edges))
            edge_attr = torch.rand(num_edges, num_edge_features)
            graph= Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

            return graph

        self.generate_pose_dict()

        # intra-person graph generation    
        for frame_id in self.pose:
            for instance in frame_id['instances']:
                self.graph_dict[(frame_id, instance)] = generate_intra_spatial_edge(self.pose_dict[(frame_id, instance)])
                # each graph size is Data(x=[14, 3], edge_index=[2, 19], edge_attr=[19, 3])

        # inter-person graph generation
        for frame_id in self.pose:
            frame_graph = None

            for instance in frame_id['instances']:
                player_graph = self.graph_dict[(frame_id, instance)]
                if frame_graph == None :
                    frame_graph = Data(x=player_graph.x.clone(),
                                       edge_index = player_graph.edge_index.clone(),
                                       edge_attr = player_graph.edge_attr.clone())
                    continue

                x_combined = torch.cat([frame_graph.x, player_graph.x], dim=0)

                # edge index modification
                player_graph_edge_offset = player_graph.edge_index + frame_graph.x.size(0)
                edge_index = torch.cat([frame_graph.edge_index, player_graph_edge_offset], dim=1)

                # merge feature
                edge_feature = torch.cat([frame_graph.edge_attr, player_graph.edge_attr], dim=0)
                
                # merge graph
                frame_graph = Data(x=x_combined, edge_index = edge_index, edge_attr =edge_feature)

            self.frame_graph_list.append(frame_graph) 
            # each graph size is Data(x=[85, 3], edge_index=[2, 95], edge_attr=[95, 3]) when k = 5
        inter_edge_index = [[],[]]
        inter_edge_feature = [ ]

        for frame_id in self.pose:
            inter_edge_index = [[],[]]
            inter_edge_feature = [ ]
            '''
            for instance in frame_id['instances']:
                for player2 in range(player1+1, self.top_player_num):
                    for joint in range(self.num_joint):
                        player1_offset = self.num_joint*player1
                        player2_offset = self.num_joint*player2
                        inter_edge_index[0].append(player1_offset+joint)
                        inter_edge_index[1].append(player2_offset+joint)
                        inter_edge_index[0].append(player2_offset+joint)
                        inter_edge_index[1].append(player1_offset+joint)
            '''
            inter_edge_index = torch.tensor(inter_edge_index)

            # [0, 0, 1] is inter_edge_feature
            inter_edge_feature = torch.tensor([0,0,1],dtype=torch.float).repeat(len(inter_edge_index[0]),1)

            frame_graph = self.frame_graph_list[frame_id]
            
            edge_index = torch.cat([frame_graph.edge_index,inter_edge_index], dim= 1)
            edge_feature = torch.cat([frame_graph.edge_attr, inter_edge_feature], dim=0)
            
            frame_graph = Data(x=frame_graph.x, edge_index=edge_index, edge_attr=edge_feature)
            self.frame_graph_list[frame_id] = frame_graph
            # each graph size is Data(x=[85, 3], edge_index=[2, 530], edge_attr=[530, 3]) when k = 5
        
        for index, graph in enumerate(self.frame_graph_list):
            visualize_graph(graph, f'results/{index}.png')

        # intra-person temporal graph generation
        clip_graph = None

        # graph merging
        for frame_id in range(len(self.pose)):
            frame_graph = self.frame_graph_list[frame_id]
            if clip_graph is None:
                clip_graph = Data(x=frame_graph.x.clone(),
                                       edge_index = frame_graph.edge_index.clone(),
                                       edge_attr = frame_graph.edge_attr.clone())
                continue
            
            x_combined = torch.cat([clip_graph.x, frame_graph.x], dim=0)
            # if you want to visualize the graph, then swap the above code into:
            # x_combined = torch.cat([clip_graph.x, frame_graph.x+500*frame_id], dim=0)

            # edge index modification
            frame_graph_edge_offset = frame_graph.edge_index + clip_graph.x.size(0)
            edge_index = torch.cat([clip_graph.edge_index, frame_graph_edge_offset], dim=1)

            # merge feature
            edge_feature = torch.cat([clip_graph.edge_attr, frame_graph.edge_attr], dim=0)
            
            # merge graph
            clip_graph = Data(x=x_combined, edge_index = edge_index, edge_attr =edge_feature)
            # the size of clip graph is Data(x=[2125, 3], edge_index=[2, 13250], edge_attr=[13250, 3]) when k = 5 


        # connect temporal edges: 
        # joints in adjacent frames are connected
        # the edge feature is defined as [0, 1, 0]

        temporal_edge_index = [[], []]
        temporal_edge_feature = []

        for frame_id in range(len(self.pose)):
            # connect joint-k*17 -- joint -- joint+k*17
            num_node_per_frame = self.top_player_num * self.num_joint
            frame_offset = frame_id * num_node_per_frame
            for node_index in range(num_node_per_frame):

                if frame_id == len(self.pose)-1: 
                    continue
                
                # connect with next frame -- undirected edge
                temporal_edge_index[0].append(node_index+frame_offset)
                temporal_edge_index[1].append(node_index+frame_offset+num_node_per_frame)
                temporal_edge_index[0].append(node_index+frame_offset+num_node_per_frame)
                temporal_edge_index[1].append(node_index+frame_offset)

        temporal_edge_index = torch.tensor(temporal_edge_index)
        temporal_edge_feature = torch.tensor([0,0,1], dtype=torch.float).repeat(len(temporal_edge_index[0]),1)
        
        edge_index = torch.cat([clip_graph.edge_index, temporal_edge_index], dim =1)
        edge_feature = torch.cat([clip_graph.edge_attr, temporal_edge_feature], dim= 0)

        clip_graph = Data(x= clip_graph.x, edge_index=edge_index, edge_attr = edge_feature)
        # the size is Data(x=[2125, 3], edge_index=[2, 17330], edge_attr=[17330, 3])

        return clip_graph


    
if __name__ == '__main__':
    data_path = 'models/clip_0.json'
    data_path2 = 'example.pkl'
    with open(data_path, 'r') as file:
        pose_data = json.load(file)

    
    pose_graph = PoseGraph(pose_data)
    pose_graph.generate_graph()

