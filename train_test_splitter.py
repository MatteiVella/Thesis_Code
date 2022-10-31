import os
import numpy as np
import shutil
# https://stackoverflow.com/questions/57394135/split-image-dataset-into-train-test-datasets

seeds = [1, 10, 100]
root_dir = r'C:\Users\matte\OneDrive\Desktop\Thesis\Thesis_Code\datasets\MalteseFood_Dataset_Final\resized\combined'
config_dir = ['/Config1', '/Config2', '/Config3']

test_ratio = 0.30
counter = 0

for cls in config_dir:
    os.makedirs(root_dir+cls +'/train/')
    os.makedirs(root_dir+cls +'/test/')

    src = root_dir + r'\TopView'

    allFileNames = os.listdir(src)
    np.random.seed(seeds[counter])
    counter += 1
    np.random.shuffle(allFileNames)
    train_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames)* (1 - test_ratio))])


    train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
    test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

    print("*****************************")
    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Testing: ', len(test_FileNames))
    print("*****************************")

    for name in train_FileNames:
            shutil.copy(name, root_dir+cls +'/train/')

    for name in test_FileNames:
            shutil.copy(name, root_dir+cls +'/test/')


print("Copying Done!")



