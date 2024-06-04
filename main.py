import os
import logging
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from SoccerNet.Evaluation.MV_FoulRecognition import evaluate

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str('6')

import torch
from datasets.dataset import MultiViewDataset
from utils.train import trainer, evaluation
import torch.nn as nn
import torchvision.transforms as transforms
from models import Model
from config.classes import EVENT_DICTIONARY, INVERSE_EVENT_DICTIONARY


def checkArguments():

    # args.num_views
    if args.num_views > 5 or  args.num_views < 1:
        print("Could not find your desired argument for --args.num_views:")
        print("Possible number of views are: 1, 2, 3, 4, 5")
        exit()

    # args.data_aug
    if args.data_aug != 'Yes' and args.data_aug != 'No':
        print("Could not find your desired argument for --args.data_aug:")
        print("Possible arguments are: Yes or No")
        exit()

    # args.weighted_loss
    if args.weighted_loss != 'Yes' and args.weighted_loss != 'asl' and args.weighted_loss != 'No':
        print("Could not find your desired argument for --args.weighted_loss:")
        print("Possible arguments are: Yes or No")
        exit()

    # args.start_frame
    if args.start_frame > 124 or  args.start_frame < 0 or args.end_frame - args.start_frame < 2:
        print("Could not find your desired argument for --args.start_frame:")
        print("Choose a number between 0 and 124 and smaller as --args.end_frame")
        exit()

    # args.end_frame
    if args.end_frame < 1 or  args.end_frame > 125:
        print("Could not find your desired argument for --args.end_frame:")
        print("Choose a number between 1 and 125 and greater as --args.start_frame")
        exit()

    # args.fps
    if args.fps > 25 or  args.fps < 1:
        print("Could not find your desired argument for --args.fps:")
        print("Possible number for the fps are between 1 and 25")
        exit()


def main(*args):

    if args:
        args = args[0]
        LR = args.LR
        gamma = args.gamma
        step_size = args.step_size
        start_frame = args.start_frame
        end_frame = args.end_frame
        weight_decay = args.weight_decay
        model_name = args.model_name
        num_views = args.num_views
        fps = args.fps
        number_of_frames = int((args.end_frame - args.start_frame) / ((args.end_frame - args.start_frame) / (((args.end_frame - args.start_frame) / 25) * args.fps)))
        batch_size = args.batch_size
        data_aug = args.data_aug
        path = args.path
        weighted_loss = args.weighted_loss
        max_num_worker = args.max_num_worker
        max_epochs = args.max_epochs
        continue_training = args.continue_training
        only_evaluation = args.only_evaluation
        path_to_model_weights = args.path_to_model_weights
        model_to_store = args.model_to_store
        multi_gpu = args.multi_gpu
    else:
        print("EXIT")
        exit()

    
    print(f'CUDA_VISIBLE_DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    print(f'Available CUDA devices: {torch.cuda.device_count()}')
    print(f'Current device index: {torch.cuda.current_device()}')

    # Logging information
    numeric_level = getattr(logging, 'INFO'.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % 'INFO')

    if model_to_store == '':
        model_to_store = 'model_store'
    best_model_path = model_to_store
    try:
        os.makedirs(best_model_path)
    except FileExistsError:
        pass

    log_path = os.path.join(best_model_path, "logging.log")

    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    # Initialize the data augmentation
    if data_aug == 'Yes':
        transformAug = transforms.Compose([
                                          transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1)),
                                          transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                                          transforms.RandomRotation(degrees=5),
                                          transforms.ColorJitter(brightness=0.5, saturation=0.5, contrast=0.5),
                                          transforms.RandomHorizontalFlip()
                                          ])

    if only_evaluation == 0:
        dataset_Test2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Test', num_views = 5)
        
        test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
            batch_size=batch_size, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
    elif only_evaluation == 1:
        dataset_Chall = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Chall', num_views = 5)

        chall_loader2 = torch.utils.data.DataLoader(dataset_Chall,
            batch_size=batch_size, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
    elif only_evaluation == 2:
        dataset_Test2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Test', num_views = 5)
        dataset_Chall = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Chall', num_views = 5)

        test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
            batch_size=batch_size, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
        
        chall_loader2 = torch.utils.data.DataLoader(dataset_Chall,
            batch_size=batch_size, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
    else:
        # Create Train Validation and Test datasets
        dataset_Train = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Train',
            num_views = num_views, transform=transformAug)
        dataset_Valid2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Valid', num_views = 5, transform=transformAug)
        dataset_Test2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Test', num_views = 5, transform=transformAug)

        # Create the dataloaders for train validation and test datasets
        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=batch_size, shuffle=True,
            num_workers=max_num_worker, pin_memory=True)

        val_loader2 = torch.utils.data.DataLoader(dataset_Valid2,
            batch_size=batch_size, shuffle=True,
            num_workers=max_num_worker, pin_memory=True)
        
        test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
            batch_size=batch_size, shuffle=True,
            num_workers=max_num_worker, pin_memory=True)
    
    ###################################
    #       LOADING THE MODEL         #
    ###################################

    model = Model()

    if multi_gpu:
        model = nn.DataParallel(model)
        model = model.cuda()
    else :
        model = model.cuda()

    if path_to_model_weights != "":
        path_model = os.path.join(path_to_model_weights)
        load = torch.load(path_model)
        print('load model...')
        model.load_state_dict(load['state_dict'])
    
    if only_evaluation == 3:

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, 
                                    betas=(0.9, 0.999), eps=1e-07, 
                                    weight_decay=weight_decay, amsgrad=False)
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        epoch_start = 0

        if continue_training:
            path_model = os.path.join(log_path, 'model.pth.tar')
            load = torch.load(path_model)
            model.load_state_dict(load['state_dict'])
            optimizer.load_state_dict(load['optimizer'])
            scheduler.load_state_dict(load['scheduler'])
            epoch_start = load['epoch']


        if weighted_loss == 'Yes':
            criterion_offence_severity = nn.CrossEntropyLoss(weight=dataset_Train.getWeights()[0].cuda())
            criterion_action = nn.CrossEntropyLoss(weight=dataset_Train.getWeights()[1].cuda())
            criterion = [criterion_offence_severity, criterion_action]
        else:
            criterion_offence_severity = nn.CrossEntropyLoss()
            criterion_action = nn.CrossEntropyLoss()
            criterion = [criterion_offence_severity, criterion_action]


    # Start training or evaluation
    if only_evaluation == 0:
        prediction_file = evaluation(
            test_loader2,
            model,
            set_name="test",
            multi_gpu=multi_gpu
        ) 
        results = evaluate(os.path.join(path, "Test", "annotations.json"), prediction_file)
        print("TEST")
        print(results)

    elif only_evaluation == 1:
        prediction_file = evaluation(
            chall_loader2,
            model,
            set_name="chall",
            multi_gpu=multi_gpu
        )

        results = evaluate(os.path.join(path, "Chall", "annotations.json"), prediction_file)
        print("CHALL")
        print(results)

    elif only_evaluation == 2:
        prediction_file = evaluation(
            test_loader2,
            model,
            set_name="test",
            multi_gpu=multi_gpu
        )

        results = evaluate(os.path.join(path, "Test", "annotations.json"), prediction_file)
        print("TEST")
        print(results)

        prediction_file = evaluation(
            chall_loader2,
            model,
            set_name="chall",
            multi_gpu=multi_gpu
        )

        results = evaluate(os.path.join(path, "Chall", "annotations.json"), prediction_file)
        print("CHALL")
        print(results)
    else:
        trainer(train_loader, val_loader2, test_loader2, model, optimizer, scheduler, criterion, 
                best_model_path, epoch_start, model_name=model_name, path_dataset=path, max_epochs=max_epochs, multi_gpu=multi_gpu)
    return 0



if __name__ == '__main__':

    parser = ArgumentParser(description='my method', formatter_class=ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--path',   required=True, type=str, help='Path to the dataset folder' )
    parser.add_argument('--max_epochs',   required=False, type=int,   default=60,     help='Maximum number of epochs' )
    parser.add_argument('--model_name',   required=False, type=str,   default="VARS",     help='named of the model to save' )
    parser.add_argument('--batch_size', required=False, type=int,   default=2,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=1e-04, help='Learning Rate' )
    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=1, help='number of worker to load data')
    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')
    parser.add_argument("--continue_training", required=False, action='store_true', help="Continue training")
    parser.add_argument("--num_views", required=False, type=int, default=5, help="Number of views")
    parser.add_argument("--data_aug", required=False, type=str, default="Yes", help="Data augmentation")
    parser.add_argument("--pre_model", required=False, type=str, default="r2plus1d_18", help="Name of the pretrained model")
    parser.add_argument("--weighted_loss", required=False, type=str, default="Yes", help="If the loss should be weighted")
    parser.add_argument("--start_frame", required=False, type=int, default=0, help="The starting frame")
    parser.add_argument("--end_frame", required=False, type=int, default=125, help="The ending frame")
    parser.add_argument("--fps", required=False, type=int, default=25, help="Number of frames per second")
    parser.add_argument("--step_size", required=False, type=int, default=3, help="StepLR parameter")
    parser.add_argument("--gamma", required=False, type=float, default=0.1, help="StepLR parameter")
    parser.add_argument("--weight_decay", required=False, type=float, default=0.001, help="Weight decacy")
    parser.add_argument("--multi_gpu", action='store_true', help="Enable multigpu mode")
    parser.add_argument("--model_to_store", required=False, type=str, default="", help="path to store the model weights")

    parser.add_argument("--only_evaluation", required=False, type=int, default=3, help="Only evaluation, 0 = on test set, 1 = on chall set, 2 = on both sets and 3 = train/valid/test")
    parser.add_argument("--path_to_model_weights", required=False, type=str, default="", help="Path to the model weights")


    args = parser.parse_args()

    ## Checking if arguments are valid
    checkArguments()

    # Setup the GPU
    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)

    # Start the main training function
    start=time.time()
    logging.info('Starting main function')
    
    main(args, False)
    logging.info(f'Total Execution Time is {time.time()-start} seconds')

