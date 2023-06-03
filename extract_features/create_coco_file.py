# %% Load Libraries
import numpy as np
from pathlib import Path
import copy
import json
from rich.progress import Progress


import kwimage
import kwcoco

import logging
from rich.logging import RichHandler

logging.basicConfig(
    level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")


# %% Set Parameters

path_to_dataset_folder = '/mnt/3bcbbdb5-8833-4d40-a51a-7fca7f39ec24/data2/xview2/geotiffs'
output_coco_file = './xview2.coco.zip'

# %% Load dataset


dataset = Path(path_to_dataset_folder)

img_list = list(dataset.glob('*/*/*.tif'))
# %%  Test to see if labels for each image

def get_labels_path(img_path):
    labels_path = copy.deepcopy(img_path)
    return Path(img_path.parent.parent,'labels',img_path.stem + '.json')

for img_path in img_list:
    labels_path = get_labels_path(img_path)
    if not labels_path.exists():
        log.error(labels_path)
    
else:
    log.info('All label files found')




# %%  Create COCO

coco = kwcoco.CocoDataset()

coco.add_category('un-classified')
coco.add_category('no-damage')
coco.add_category('minor-damage')
coco.add_category('major-damage')
coco.add_category('destroyed')

n = len(img_list)

with Progress() as progress:
    task1 = progress.add_task("[red]running_image...", total=n)

    for img_path in img_list:
        progress.update(task1, advance=1, refresh=True)

        # Load labels
        labels_path = get_labels_path(img_path)
        with open(labels_path,'r') as f:
            label_json = json.load(f)


        # Collect Metadata 
        metadata = label_json['metadata']
        pre_disaster = labels_path.name[-17:-14] == 'pre'
        if pre_disaster:
            time = 'pre_disaster'
        else:
            time = 'post disaster'

        # add image to coco
        gid = coco.add_image(img_path, width=metadata['width'], 
                            height=metadata['height'], metadata=metadata,
                            split=img_path.parts[-3], time=time)

        # For each annotation, extract info and add to coco
        for d_ann in label_json['features']['xy']:
            properties = d_ann['properties']
            polygon = d_ann['wkt']
            bbox = kwimage.Polygon.coerce(d_ann['wkt']).to_box().toformat('xywh')

            #pre disaster doesn't have labels so assuming no-damage
            if pre_disaster:
                subtype = 'no-damage'
            else:
                subtype = properties['subtype']

            cid = coco.name_to_cat[subtype]['id']
            coco.add_annotation(image_id=gid, category_id=cid, 
                                bbox=bbox, polygon=polygon, properties=properties)


# %%  Save Dataset

# coco._ensure_json_serializable()

coco.fpath = output_coco_file
coco.dump()

# %%
