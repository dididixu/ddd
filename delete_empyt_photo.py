import os
import shutil

root_dir = r'D:\dataset\merge-dataset'
photo_url = os.path.join(root_dir, 'JPEGImages')
xml_url = os.path.join(root_dir, 'Annotations')
xml_url_copy = os.path.join(root_dir, 'Annotations-clear')
img = os.listdir(photo_url)
xml = os.listdir(xml_url)
for i in img:
    s = xml.index(i.split('.')[0] + '.xml')
    # print(i, img[s])
    shutil.copy(os.path.join(xml_url, xml[s]), xml_url_copy)
