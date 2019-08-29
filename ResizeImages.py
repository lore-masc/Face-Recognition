from PIL import Image
import os, sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# path = "data\lfw/"
# dirs = os.listdir( path )

# def resize():
#     for item in dirs:
#         if os.path.isfile(path+item):
#             im = Image.open(path+item)
#             f, e = os.path.splitext(path+item)
#             imResize = im.resize((228,228), Image.ANTIALIAS)
#             imResize.save(f + ' resized.jpg', 'JPEG', quality=90)

# resize()

i = 0
os.chdir('data/lfw')
all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
# print(os.chdir('data\lfw'))
for directories in all_subdirs:
    i += 1
    filelist = os.listdir(directories)
    for image in filelist[:]:
        im = Image.open("{}\{}".format(directories, image))
        imResize = im.resize((228, 228), Image.ANTIALIAS)
        imResize.save(str(i) + '.jpg', 'JPEG', quality=90)
    print(directories)