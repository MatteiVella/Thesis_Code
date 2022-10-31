import json

# Opening JSON file

f = open('C:/Users/matte/OneDrive/Desktop/Thesis/Thesis_Code/datasets/MalteseFood_Dataset_Final/resized/combined/TopView/annotation_coco.json')

# returns JSON object as
# a dictionary
data_set = json.load(f)
jsonFile = {}
listOfFoodItems = []
listOfFileNames = {}
listOfCoOrdinates = {}
isFirstTime = 0


# Getting list of food names/image names for each food/image id. The -1 is there to prevent array out of bounds error
def get_food_class(category_id):
    return listOfFoodItems[category_id-1]


def get_image_name(image_id):
    return listOfFileNames.get(image_id)


# Getting all the categories
for food in data_set['categories']:
    listOfFoodItems.append(food['name'])

# Getting all the names
for fileName in data_set['images']:
    listOfFileNames[fileName['id']] = fileName['file_name']

# Creating the list of all file names found in the json file. I am also using the get_image_name to replace the id's
for files in listOfFileNames:
    jsonFile[get_image_name(files)] = []
currentImageId = 1

# Building the JSON file in the proper format
for anno in data_set['annotations']:
    # First we need to build each object/list individually and then append them to each other
    testObject = {}
    boundingRegion = {}
    segmentation = [anno['segmentation'][0]]
    boundingRegion['BR'] = segmentation[0]
    testObject[anno['category_id']] = boundingRegion
    # Replacing the key value with the name of it rather then the id
    testObject[get_food_class(anno['category_id'])] = testObject.pop(anno['category_id'])

    if currentImageId == anno['image_id']:
        jsonFile[get_image_name(currentImageId)].append(
            testObject
        )
    else:
        currentImageId = currentImageId + 1
        jsonFile[get_image_name(currentImageId)].append(
            testObject
        )


with open('C:/Users/matte/OneDrive/Desktop/Thesis/Thesis_Code/datasets/MalteseFood_Dataset_Final/resized/combined/TopView/annotation.json', 'w', encoding='utf-8') as f:
    json.dump(jsonFile, f, ensure_ascii=False, indent=4)
#print(jsonFile)
