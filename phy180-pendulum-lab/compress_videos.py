# Compress large video files taken from a phone camera
# to smaller videos used for tracking (and to be uploaded to GitHub).

import os
import subprocess

videos = os.listdir("raw-videos/")
videos.sort()

lengths = [80, 70, 60, 50, 40, 30, 20]
filenames = [f'videos/{l}.mp4' for l in lengths]

for (video, filename) in zip(videos, filenames):
    print(video, filename)
    subprocess.run([
        'ffmpeg',
        '-i', 'raw-videos/'+video,
        '-vcodec', 'libx264',
        '-crf', '28',
        '-an', filename
    ])
    #break
