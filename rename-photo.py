import os

# ROOT_DIR = os.path.abspath(r"D:\dataset-badge")
# img_path = os.path.join(ROOT_DIR, "JPEGImages")
from PIL import Image
import cv2 as cv
import os


def PNG_JPG(PngPath):
    img = cv.imread(PngPath, 0)
    w, h = img.shape[::-1]
    infile = PngPath
    outfile = os.path.splitext(infile)[0] + ".jpg"
    img = Image.open(infile)
    img = img.resize((int(w / 2), int(h / 2)), Image.ANTIALIAS)
    try:
        if len(img.split()) == 4:
            # prevent IOError: cannot write mode RGBA as BMP
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            img.convert('RGB').save(outfile, quality=70)
            os.remove(PngPath)
        else:
            img.convert('RGB').save(outfile, quality=70)
            os.remove(PngPath)
        return outfile
    except Exception as e:
        print("PNG转换JPG 错误", e)


ROOT_DIR = os.path.abspath(r"D:\dataset\dataset-clothes\1229-final")
ROOT_TO_DIR = r"D:\dataset\dataset-clothes\1229-final"
#
save_xml = 1
# 0 不保存xml
# 1 保存xml
img_path = os.path.join(ROOT_DIR, "JPEGImages1000")
xml_path = os.path.join(ROOT_DIR, "Annotations1000")
img_to_path = os.path.join(ROOT_TO_DIR, "JPEGImages")
xml_to_path = os.path.join(ROOT_TO_DIR, "Annotations")
if os.path.isdir(img_to_path) == 0:
    os.mkdir(img_to_path)
if os.path.isdir(xml_to_path) == 0 and save_xml:
    os.mkdir(xml_to_path)

imglist = os.listdir(img_path)
i = len(os.listdir(img_to_path))
for img in imglist:
    i += 1
    if img.endswith('.jpg') or img.endswith('.JPG') or img.endswith('.jpeg') or img.endswith('.JPEG') or img.endswith('.png') or img.endswith('.PNG'):
        # if img.endswith('.png') or img.endswith('.PNG'):
        #     PNG_JPG(os.path.join(img_path, img))
        #     img = img[0:-4] + '.jpg'

        src_jpg = os.path.join(img_path, img)  # 原先的图片名字
        if img.endswith('.jpg') or img.endswith('.JPG') or img.endswith('.jpeg') or img.endswith('.JPEG'):
            # dst = os.path.join(img_path, '{:0>8d}'.format(i) + '.jpg')
            dst_jpg = os.path.join(img_to_path, '{:0>8d}'.format(i) + '.jpg')  # 根据自己的需要重新命名,可以把'E_' + img改成你想要的名字
        if img.endswith('.png') or img.endswith('.PNG'):
            # dst = os.path.join(img_path, '{:0>8d}'.format(i) + '.png')
            dst_jpg = os.path.join(img_to_path, '{:0>8d}'.format(i) + '.png')  # 根据自己的需要重新命名,可以把'E_' + img改成你想要的名字
        # os.rename(src_jpg, dst)  # 重命名,覆盖原先的名字
        if save_xml:
            if img.endswith('.jpg') or img.endswith('.JPG') or img.endswith('.png') or img.endswith('.PNG'):
                src_xml = os.path.join(xml_path, img[0:-4] + '.xml')  # 原先的xml名字
            elif img.endswith('.jpeg') or img.endswith('.JPEG'):
                src_xml = os.path.join(xml_path, img[0:-5] + '.xml')  # 原先的xml名字
            else:
                continue
            if os.path.isfile(src_xml) == 0:
                i -= 1
                continue
            dst_xml = os.path.join(xml_to_path, '{:0>8d}'.format(i) + '.xml')
            os.system("copy %s %s" % (src_xml, dst_xml))
        # print(src_jpg, dst_jpg)
        # print(src_xml, dst_xml)
        os.system("""copy "%s" "%s" """ % (src_jpg, dst_jpg))

