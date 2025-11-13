import os
# coffee
# oatmeal
# pinwheels
# quesadilla
# tea
# task = 'tea'

# filenames = os.listdir('data/EgoPER/%s/trim_videos'%(task))

# print(f"len(filenames):{len(filenames)}")

# for filename in filenames:
#     id = filename[:-4]
#     if not os.path.exists('data/EgoPER/%s/frames_10fps/%s'%(task, id)):
#         os.mkdir('data/EgoPER/%s/frames_10fps/%s'%(task, id))
#     os.system('ffmpeg -i data/EgoPER/%s/trim_videos/%s.mp4 '%(task, id) + '-vf "fps=10" data/EgoPER/%s/frames_10fps/%s'%(task, id) + '/%06d.png')



# filenames = os.listdir('./data/HoloAssist/Videos')
# print(f"len(filenames):{len(filenames)}")

# for filename in filenames:
#     id = filename[:-4]
#     if not os.path.exists('./data/HoloAssist/frames_10fps/%s'%(id)):
#         os.mkdir('./data/HoloAssist/frames_10fps/%s'%(id))
#     os.system('ffmpeg -i ./data/HoloAssist/Videos/%s.mp4 '%(id) + '-vf "fps=10" ./data/HoloAssist/frames_10fps/%s'%(id) + '/%06d.png')

filenames = os.listdir('./data/Epic-Tent/Videos')
print(f"len(filenames):{len(filenames)}")

for filename in filenames:
    id = filename[:-4]
    if not os.path.exists('./data/Epic-Tent/frames_10fps/%s'%(id)):
        os.mkdir('./data/Epic-Tent/frames_10fps/%s'%(id))
    os.system('ffmpeg -i ./data/Epic-Tent/Videos/%s.mp4 '%(id) + '-vf "fps=10" ./data/Epic-Tent/frames_10fps/%s'%(id) + '/%06d.png')