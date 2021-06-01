# coding: utf-8
import os
import piexif
import exifread



def parse_image(path):
    """解析单张图片的信息"""
    file = open(path, "rb")
    tags = exifread.process_file(file)
    for tag in tags.keys():
        if tag not in ('JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote'):
            print("Key: %s, value %s" % (tag, tags[tag]))
            # info.ImageWidth = tags["Image ImageWidth"]
            # info.ImageLength = tags["Image ImageLength"]
            # info.Make = tags["Image Make"]
            # info.Model = tags["Image Model"]
            # info.GPSLatitudeRef = tags["GPS GPSLatitudeRef"]
            # info.GPSLatitude = tags["GPS GPSLatitude"]
            # info.GPSLongitudeRef = tags["GPS GPSLongitudeRef"]
            # info.GPSLongitude = tags["GPS GPSLongitude"]
            # info.DateTimeOriginal = tags["EXIF DateTimeOriginal"]


def earse_exif(dir):
    for root, dirs, files in os.walk(dir):
        for name in files:
            file = os.path.join(root, name)
            if file.endswith("jpg") or file.endswith("jpeg"):
                print(name)
                piexif.remove(file)
                # parse_image(file)

if __name__ == '__main__':
    dir = r"D:\dataset\dataset-clothes\1229-final\JPEGImages"
    earse_exif(dir)