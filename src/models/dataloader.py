

"""
Custom Video Dataloader
Author/Maintainer: Amil Khan

Loads the labels from different Deepfake folders and returns balanced classes.

"""


import numpy as np
import pandas as pd
import os
import glob
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


def labls(root_path, num_folders=1):

    num   = 0  
    reals = []
    fakes = []
    while num != num_folders:
        jsoon = glob.glob('{}{}'.format(root_path, num)+"/metadata.json")
        files = glob.glob("{}{}/*.mp4".format(root_path, num))
        df    = pd.read_json(jsoon[0])
        for k in files:
        #     print(k)
            base = os.path.basename(k)
            if os.path.splitext(base)[0]+'.mp4' in df.columns:
                labl = df[os.path.splitext(base)[0]+'.mp4'][0]
#                 print(k,labl)
                if labl == 'FAKE':
#                     labl = 1
                    fakes.append((k, 1))
                elif labl == 'REAL':
                    reals.append((k, 0))
        num+=1    

    print("""-------------------------------------------------------------
---> Successfully Loaded Labels!

Real Videos     Fakes Videos     Folders Loaded    
============    ============     ==============
    {}              {}               {}
    
NOTE: For balanced classes, we are using {} from each class.
-------------------------------------------------------------
          """.format(len(reals), len(fakes), num_folders, len(reals)))        
    return reals + fakes[:len(reals)]










class DeepFakeDataset(Dataset):
    """
    
    +----------------------------------------------------------------------------------+    
    
        Deep Fake Challenge Dataset
        Author/Maintainer: Amil Khan


        Specifically made for the Facebook DeepFake Challenge 2020, this function

        - Loads all filepaths of every video given the Challenge folder structure
        - Parses the JSON for labels and matches them to its correct video file
        - Extracts VIDEO and AUDIO seperately from each file
        - Returns the video, audio, and label information with equal number of
          REAL videos and equal number of FAKE videos

        
        Parameters:
        -----------
        root_dir : String 
        Directory with all the videos.
        
        num_frames : Int
        Number of frames to load from the video, default=100


        Returns
        -------
        video : Tensor
        The video frames with Type Tensor with shape  FRAMES X HEIGHT X WIDTH X CHANNEL

        audio : Tensor
        The audio frames with Type Tensor with shape  CHANNEL X NUMBER OF POINTS

        label : Tensor
        The label of the sample, where 0 is REAL and 1 is FAKE
    
    +----------------------------------------------------------------------------------+
    """

    def __init__(self, root_dir, num_frames=50, num_folders=1):
        """Access all of the videos and labels, root directory, number of frames, 
        number of folders, and length of the dataset"""
        self.videos      = pd.DataFrame(labls(root_dir,num_folders))
        self.root_dir    = root_dir
        self.num_vids    = len(self.videos)
        self.num_frames  = num_frames
        self.num_folders = num_folders
        
    
    def video_reader(self, video_file):
        """
        +----------------------------------------------------------------------------------+
        
        This function reads the videos and returns a tensor of the video data
        with shape FRAMES X HEIGHT X WIDTH X CHANNEL.

        You can probably adapt this generic function to read in various types of 
        video data quickly. You can also call this as a standalone function to 
        view your data for rapid prototyping.


        Parameters:
        -----------
        video_file :  Path to video       


        Returns
        -------
        video : Tensor
        The video frames with Type Tensor with shape  FRAMES X HEIGHT X WIDTH X CHANNEL       

        +----------------------------------------------------------------------------------+
        """
        vid = torchvision.io.read_video('{}'.format(video_file), pts_unit='sec')
        vid = list(vid)
        if 720 in vid[0][0].shape:
            vid[0] = F.upsample(vid[0][None,].type(torch.float).permute(0,1,4,2,3), size=(1080,1920,3))[0]
            print('WARNING: Upsampled')
        if 1440 in vid[0][0].shape:
            print('WARNING: Cropping Image')
            vid[0] = F.interpolate(vid[0][None,].type(torch.float).permute(0,1,4,2,3), size=(1080,1920,3))[0]
        if vid[0][0].shape != (1080, 1920, 3):
            vid[0] = vid[0].permute(0,2,1,3)
        return tuple(vid) 
    
    
    def __len__(self):
        return len(self.videos) 
    
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist() 
#         print(idx)
        videodata  = self.videos.iloc[idx]
        vid_aud = self.video_reader(videodata[0])
        vid = (vid_aud[0][:self.num_frames].type(torch.float) - vid_aud[0][:self.num_frames].type(torch.float).mean()) /  (vid_aud[0][:self.num_frames].type(torch.float).max()- vid_aud[0][:self.num_frames].type(torch.float).mean())
#         print('Audio', idx, vid_aud[1].shape) 'audio':vid_aud[1][:,:400320].type(torch.float), 
        return {'video':vid.permute(3,0,1,2), 'label':torch.tensor(videodata[1]).type(torch.float)}