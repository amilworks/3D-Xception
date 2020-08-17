
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
plt.rcParams['figure.dpi'] = 150



def video_visualizer(video_path, frame_num=1):
    """
    Visualize single frames in a video
    
    
    Parameters
    ----------
    
    video_path: String 
    Path to the video 
    
    frame_num: Int
    Frame to visualize
    
    """
    
    vid = torchvision.io.read_video(video_path, pts_unit='sec')[0]
    print("Video Shape: {}".format(vid.shape))
    plt.imshow(vid[frame_num])
    plt.title("DeepFake Dataset: Example Frame {}\nSize: {} x {}".format(frame_num, vid.shape[1], vid.shape[2]), loc='left', fontsize=8, pad=5)
    plt.axis("off")