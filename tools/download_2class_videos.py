import os
from torchvision.datasets.utils import download_url

os.makedirs('videos', exist_ok=True)

video_url = 'https://github.com/dai-ichiro/robo-one/raw/main/video_1.mp4'
video_fname = 'target.mp4'
download_url(video_url, root = 'videos', filename = video_fname)

video_url = 'https://github.com/dai-ichiro/robo-one/raw/main/video_2.mp4'
video_fname = 'non_target.mp4'
download_url(video_url, root = 'videos', filename = video_fname)
