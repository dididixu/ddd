import os
from shutil import copyfile

path = r"D:\dataset-workcard\VOC-Card-clear\JPEGImages"
xml_path = r"D:\dataset-workcard\VOC-Card-clear\Annotations-temp"
xml_copy_path = r"D:\dataset-workcard\VOC-Card-clear\Annotations"
s = os.listdir(path)
for i in s:
    name = i[:-4]
    path_source = os.path.join(xml_path, name + '.xml')
    path_to = os.path.join(xml_copy_path, name + '.xml')
    if os.path.exists(path_source):
        # copyfile(path_source, xml_copy_path)
        os.system("copy %s %s" % (path_source, path_to))
        # print("True")
    else:
        print("False")
