import os

xml_path = r"D:\dataset\competition-trash\xml"
ROOT_DIR = os.path.abspath(r"D:\dataset\competition-trash\pre-datasets")
ROOT_TO_DIR = r"D:\dataset\competition-trash\clear"
#
img_path = os.path.join(ROOT_DIR, "JPEGImages")
xml_path = os.path.join(ROOT_DIR, "Annotations")
img_to_path = os.path.join(ROOT_TO_DIR, "JPEGImages")
xml_to_path = os.path.join(ROOT_TO_DIR, "Annotations")
if os.path.isdir(img_to_path) == 0:
    os.mkdir(img_to_path)
if os.path.isdir(xml_to_path) == 0:
    os.mkdir(xml_to_path)

for root, dirs, files in os.walk(xml_path):
    dirs_te = root.split('\\')[-1]
    if len(dirs_te) <= 2:
        file = os.listdir(root)
        for img in imglist:
            i += 1
            if img.endswith('.jpg') or img.endswith('.JPG') or img.endswith('.jpeg') or img.endswith('.JPEG'):
                src_jpg = os.path.join(img_path, img)  # ԭ�ȵ�ͼƬ����
                dst_jpg = os.path.join(img_to_path, '{:0>8d}'.format(i) + '.jpg')  # �����Լ�����Ҫ��������,���԰�'E_' + img�ĳ�����Ҫ������
                # os.rename(src, dst)  # ������,����ԭ�ȵ�����
                if img.endswith('.jpg') or img.endswith('.JPG'):
                    src_xml = os.path.join(xml_path, img[0:-4] + '.xml')  # ԭ�ȵ�xml����
                elif img.endswith('.jpeg') or img.endswith('.JPEG'):
                    src_xml = os.path.join(xml_path, img[0:-5] + '.xml')  # ԭ�ȵ�xml����
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
