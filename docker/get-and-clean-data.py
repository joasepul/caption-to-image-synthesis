import cv2
import glob
import numpy as np
import math
import os


# this selects the first 1975 images, change the regex if you want more
img_paths = glob.glob('/coco2017/train2017/*.jpg')
training_imgs_num = len(img_paths)
print(f'loaded a total of {training_imgs_num} imgs.')


print(f'Cleaning and saving {training_imgs_num} imgs to /coco2017/cleaned-data/ ...')
os.mkdir( '/coco2017/cleaned-data') if not os.path.exists('/coco2017/cleaned-data') else None

for img_path in img_paths:
    img = cv2.imread(img_path)
    y,x,_ = img.shape #(y, x) not (x, y)
    
    margin = abs(y-x)/2
    if x > y:        
        # Image is tall
        img = img[:,int(math.floor(margin)):int(math.floor(x-margin))]
    elif y > x:
        # Image is wide
        img = img[int(math.floor(margin)):int(math.floor(y-margin)),:]

    if (img.shape[0] != img.shape[1]):
        print('Dim mismatch')
        
    img = cv2.resize(img, (128,128))
    cv2.imwrite(os.path.join('/coco2017/cleaned-data', img_path.split('/')[-1]), img)
print("Done cleaning and saving.")


# category_mapping = {}

# numeric_mapping = { 'person' : 1, 'vehicle' : 2, 'outdoor' : 3, 'animal' : 4, 'accessory' : 5, 'sports' : 6, \
# 'kitchen' : 7, 'food' : 8, 'furniture' : 9, 'electronic' : 10, 'appliance' : 11, 'indoor' : 12 }

# for dic in instances_df.categories:
#     category_mapping[dic['id']] = numeric_mapping[dic['supercategory']]

# def getSuperCat(cat):
#     return category_mapping[cat]
    

def multihot_encode(data, num_classes=12, clip=False):
    mhe, _ = np.histogram(data,bins=num_classes,range=(0,num_classes-1))
    if clip:
        return np.clip(mhe,0,1)
    
    mhe = [x/sum(mhe) if sum(mhe) > 0 else x/1.0 for x in mhe]
    return mhe
vec_multihot_encode = np.vectorize(multihot_encode)


import pandas as pd

def clean_caption(cap):
    cap = cap.replace('.', ' ')
    cap = cap + "."
    return cap
    

print("Generating caption/categories csv...")
with open('/coco2017/annotations/captions_train2017.json') as annot_file:
    captions_df = pd.read_json(annot_file, typ='series')
    annot_df = pd.DataFrame(data=captions_df['annotations'])
    annot_df = annot_df.astype({'image_id': 'int32'})
    annot_df.sort_values(by=['image_id'], axis=0, inplace=True)
    annot_df['caption'] = annot_df['caption'].apply(clean_caption)
    annot_df = annot_df.groupby('image_id')['image_id', 'caption'].agg('|'.join)
    

    with open('/coco2017/annotations/instances_train2017.json') as instance_file:
        instances_df = pd.read_json(instance_file, typ='series')
        
        category_mapping = {}

        numeric_mapping = { 'person' : 1, 'vehicle' : 2, 'outdoor' : 3, 'animal' : 4, 'accessory' : 5, 'sports' : 6,         'kitchen' : 7, 'food' : 8, 'furniture' : 9, 'electronic' : 10, 'appliance' : 11, 'indoor' : 12 }

        for dic in instances_df.categories:
            category_mapping[dic['id']] = numeric_mapping[dic['supercategory']]
        
        image_cats = {i : [] for i in list(annot_df.index)}
        image_supercats = {i : [] for i in list(annot_df.index)}
        
        for row in instances_df.annotations:
            image_cats[row['image_id']].append(row['category_id'])
            image_supercats[row['image_id']].append(category_mapping[row['category_id']])
        
        annot_df.insert(1, 'categories', image_cats.values())
        annot_df.insert(2, 'super_categories', list(map(multihot_encode, list(image_supercats.values()))))

print("Generated.")

print("Saving file coco-captions-with-categories.csv...")
annot_df.to_csv('/coco2017/coco-captions-with-categories.csv')
print("Saved.")

print("Data cleaning done.")

