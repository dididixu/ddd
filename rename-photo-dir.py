import os

xml_path = r"D:\dataset\competition-trash\xml"
photo_path = r"D:\dataset\competition-trash\photo"
ROOT_DIR = os.path.abspath(r"D:\dataset\competition-trash\pre-datasets")
ROOT_TO_DIR = r"D:\dataset\competition-trash\clear"
#
img_to_path = os.path.join(ROOT_TO_DIR, "JPEGImages")
xml_to_path = os.path.join(ROOT_TO_DIR, "Annotations")
if os.path.isdir(img_to_path) == 0:
    os.mkdir(img_to_path)
if os.path.isdir(xml_to_path) == 0:
    os.mkdir(xml_to_path)

imglist = os.listdir(img_to_path)
i = len(imglist)

for root, dirs, files in os.walk(photo_path):
    dirs_te = root.split('\\')[-1]
    xml_dirs_te = dirs_te.split('_')[-1]
    if len(xml_dirs_te) <= 2:
        imglist = os.listdir(root)
        for img in imglist:
            i += 1
            if img.endswith('.jpg') or img.endswith('.JPG') or img.endswith('.jpeg') or img.endswith('.JPEG'):
                src_jpg = os.path.join(photo_path, dirs_te, img)  # 原先的图片名字
                dst_jpg = os.path.join(img_to_path, '{:0>8d}'.format(i) + '.jpg')  # 根据自己的需要重新命名,可以把'E_' + img改成你想要的名字
                # os.rename(src, dst)  # 重命名,覆盖原先的名字
                if img.endswith('.jpg') or img.endswith('.JPG'):
                    src_xml = os.path.join(xml_path, xml_dirs_te, img[0:-4] + '.xml')  # 原先的xml名字
                elif img.endswith('.jpeg') or img.endswith('.JPEG'):
                    src_xml = os.path.join(xml_path, xml_dirs_te, img[0:-5] + '.xml')  # 原先的xml名字
                else:
                    continue
                if os.path.isfile(src_xml) == 0:
                    i -= 1
                    continue
                dst_xml = os.path.join(xml_to_path, '{:0>8d}'.format(i) + '.xml')
                # print(src_jpg, dst_jpg)
                # print(src_xml, dst_xml)
                os.system("copy %s %s" % (src_jpg, dst_jpg))
                os.system("copy %s %s" % (src_xml, dst_xml))
