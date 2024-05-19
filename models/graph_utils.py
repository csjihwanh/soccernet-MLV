from torchvision.io.video import read_video
import torch
import pickle
import numpy as np
import torch_geometric
from torch_geometric.data import Data
import networkx as nx 
import matplotlib.pyplot as plt


#from ..datasets.dataset import MultiViewDataset
import torch

def generate_intra_spatial_edge(pose):
    """
    convert pose of one person to connected graph
    pose format should be match to CrowdPose(14 keypoints)

    args: coco pose 2d list
    
    return: connected torch geometric graph object
    
    # Keypoint definition 

    keypoint definition:
        0: left shoulder
        1: right shoulder
        2: left elbow
        3: right elbow
        4: left wrist
        5: right wrist
        6: left hip
        7: right hip
        8: left knee
        9: right knee
        10: left ankle
        11: right ankle
        12: top head
        13: neck
    

    # Skeleton definition
    skeleton: [
        [10, 8], [8, 6], [11, 9], [9, 7], [6, 7], 
        [0, 6], [1, 7], [0, 1], [0, 2], [1, 3], 
        [2, 4], [3, 5], [12, 13], [1, 13], [0, 13]
    ]
    """


    # Edge index for torch tensor
    # Skeleton definition
    """
    skeleton: [
        [10, 8], [8, 6], [11, 9], [9, 7], [6, 7], 
        [0, 6], [1, 7], [0, 1], [0, 2], [1, 3], 
        [2, 4], [3, 5], [12, 13], [1, 13], [0, 13]
    ]
    """

    # Edge index for torch tensor
    edge_index = torch.tensor([
        [10,  8, 11,  9,  6,  0,  1,  0,  0,  1,  2,  3, 12,  1,  0],  # Sources
        [ 8,  6,  9,  7,  7,  6,  7,  1,  2,  3,  4,  5, 13, 13, 13]   # Targets
    ], dtype=torch.long)
    
    # to make undirected edges
    edge_index_backward = torch.stack([edge_index[1], edge_index[0]], axis=1).transpose(0,1)
    edge_index = torch.cat([edge_index, edge_index_backward],axis=1)

    num_edges = edge_index.size()[1]

    # constant weight edges 
    edge_weight = torch.tensor(1.0,dtype=torch.float).repeat(num_edges,1) 

    keypoints = torch.tensor(pose)
    keypoints_num = torch.arange(14).unsqueeze(dim=1) # add the label of each keypoints
    keypoints = torch.cat([keypoints, keypoints_num], dim=1)
    data = Data(x=keypoints, edge_index = edge_index, edge_weight=edge_weight)
    return data

def generate_inter_spatial_edge(graphs: list):
    """
    inter-person graph generation
    
    args: list of torch.geometric graphs 
    
    return: graph of a frame 
    """
    frame_graph = None

    for graph in graphs:
        if frame_graph == None :
            frame_graph = Data(x=graph.x.clone(),
                                edge_index = graph.edge_index.clone(),
                                edge_weight = graph.edge_weight.clone())
            continue

        x_combined = torch.cat([frame_graph.x, graph.x], dim=0)

        # edge index modification
        graph_edge_offset = graph.edge_index + frame_graph.x.size(0)
        edge_index = torch.cat([frame_graph.edge_index, graph_edge_offset], dim=1)

        # merge feature
        edge_weight = torch.cat([frame_graph.edge_weight, graph.edge_weight], dim=0)
        
        # merge graph
        frame_graph = Data(x=x_combined, edge_index = edge_index, edge_weight =edge_weight)

    return frame_graph
    frame_graph_list.append(frame_graph) 
    # each graph size is Data(x=[85, 3], edge_index=[2, 95], edge_attr=[95, 3]) when k = 5

    for frame_id in range(len(self.pose)):
        inter_edge_index = [[],[]]
        inter_edge_feature = [ ]

        for player1 in range (self.top_player_num):
            for player2 in range(player1+1, self.top_player_num):
                for joint in range(self.num_joint):
                    player1_offset = self.num_joint*player1
                    player2_offset = self.num_joint*player2
                    inter_edge_index[0].append(player1_offset+joint)
                    inter_edge_index[1].append(player2_offset+joint)
                    inter_edge_index[0].append(player2_offset+joint)
                    inter_edge_index[1].append(player1_offset+joint)

        inter_edge_index = torch.tensor(inter_edge_index)

        # [0, 0, 1] is inter_edge_feature
        inter_edge_feature = torch.tensor([0,0,1],dtype=torch.float).repeat(len(inter_edge_index[0]),1)

        frame_graph = self.frame_graph_list[frame_id]
        
        edge_index = torch.cat([frame_graph.edge_index,inter_edge_index], dim= 1)
        edge_feature = torch.cat([frame_graph.edge_attr, inter_edge_feature], dim=0)
        
        frame_graph = Data(x=frame_graph.x, edge_index=edge_index, edge_attr=edge_feature)
        self.frame_graph_list[frame_id] = frame_graph
        # each graph size is Data(x=[85, 3], edge_index=[2, 530], edge_attr=[530, 3]) when k = 5


def visualize_graph(graph,filename):
    #view_fixing_node = torch.tensor([[0,0],[1000,1000]], dtype=torch.float)
    #x = torch.cat([graph.x, view_fixing_node], dim=0)
    graph = Data(graph.x, graph.edge_index)

    g = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    pos = {i: (graph.x[i][0].item(), 1000-graph.x[i][1].item()) for i in range(len(graph.x))}
    plt.figure(figsize=(100, 100))
    nx.draw(g, pos)
    plt.savefig(filename)
    #plt.show()

def main():
    path = '/hub_data1/intern/SoccerNet1/mvfouls/Test/action_0/clip_1.mp4'
    start_frame= 63 
    end_frame =87 
    fps =17 

    video, _, _ = read_video(path, output_format="THWC")

    #dataset_Train = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Train',
    #    num_views = num_views, transform=transformAug, transform_model=transforms_model)
    #dataset_Valid2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Valid', num_views = 5, 
    #    transform_model=transforms_model)
    #dataset_Test2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Test', num_views = 5, 
    #    transform_model=transforms_model)

    # Create the dataloaders for train validation and test datasets
    #train_loader = torch.utils.data.DataLoader(dataset_Train,
    #    batch_size=batch_size, shuffle=True,
    #    num_workers=max_num_worker, pin_memory=True)

    #val_loader2 = torch.utils.data.DataLoader(dataset_Valid2,
    #    batch_size=1, shuffle=False,
    #    num_workers=max_num_worker, pin_memory=True)

    #test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
    #    batch_size=1, shuffle=False,
    #    num_workers=max_num_worker, pin_memory=True)


if __name__ == '__main__':
    main()