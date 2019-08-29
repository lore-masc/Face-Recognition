import os

os.chdir('data/separatePhotos/Angelo1')
i = 587
# for filename in os.listdir(path):
#   os.rename(filename, 'captured' + str(i)+'.png')
#   i = i +1

for filename in os.listdir():
  fileTitle, file_ext = os.path.splitext(filename)
  newname = '{}my-image.png'.format(i)
  i += 1
  os.rename(filename, newname)
