import os
p_path = '/scratch4/keisaito/visda/train'
dir_list = os.listdir(p_path)
write_name = open('/scratch4/keisaito/visda/all_images_train.txt','w')
for direc in dir_list:
    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path,direc))
        for file in files:
            class_name = direc
            #if class_name == 'motorcycle':
            #    class_name = 'motorbike'
            #if class_name == 'plant':
            #    class_name = 'pottedplant'
            file_name = os.path.join(p_path,direc,file)
            write_name.write('%s %s\n'%(file_name,class_name))

