from PIL import Image
import os

os.chdir('data/Grayscale')
all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]

for directories in all_subdirs:
    filelist = os.listdir(directories)
    for image in filelist[:]:
        img = Image.open("{}\{}".format(directories, image)).convert('LA')
        image = os.path.join(directories, image)
        img.save(image)

