import os
import xml.etree.ElementTree as ET
import sys
argvs = sys.argv

def load_image_set_index(ref):
    """
    Load the indexes listed in this dataset's image set file.
    """
    # Example path to image set file:
    # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
    image_set_file = os.path.join(ref)
    assert os.path.exists(image_set_file), \
        'Path does not exist: {}'.format(image_set_file)
    with open(image_set_file) as f:
        image_index = [x.strip() for x in f.readlines()]
    return image_index

def load_pascal_annotation(ref_path, index):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    filename = os.path.join(ref_path, 'Annotations', index + '.xml')
    tree = ET.parse(filename)
    objs = tree.findall('object')
    obj_list = []
    for ix, obj in enumerate(objs):
        cls = obj.find('name').text.lower().strip()
        obj_list.append(cls)
    return list(set(obj_list))

indexes = load_image_set_index(argvs[1])
images_list = open(argvs[3],'w')
for index in indexes:
    objs = load_pascal_annotation(argvs[2],index)
    write_word = os.path.join('/research/masaito/detection_dataset/VOCdevkit/VOC2007/JPEGImages', index + '.jpg' + ' ')
    for name in objs:
        write_word = write_word + name + ' '
    images_list.write(write_word + '\n')

