# %% Load Libraries
import numpy as np

import kwimage
import kwcoco

import timm 
import torch
from  torch.cuda.amp import autocast
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn
import logging
from rich.logging import RichHandler


logging.basicConfig(
    level="NOTSET", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("rich")

# %% Set Parameters


# Model to extract out the files
timm_model_name = 'convnext_xlarge_in22ft1k'

# Coco file you want to extract feature out of
input_coco_filename = './xview2.coco.json'

# Coco file you want the features to be saved into (will retain other images)
output_coco_filename = f'/mnt/3bcbbdb5-8833-4d40-a51a-7fca7f39ec24/data2/xview2/xview2_{timm_model_name}.coco.json'

device = 'cuda'


# %% Load the data


coco = kwcoco.CocoDataset.coerce(input_coco_filename)


# %%  Setup Feature Extractor

# Check to see if model is available
avail_pretrained_models = timm.list_models(pretrained=True)

if timm_model_name not in avail_pretrained_models:
    log.error(f'Model called "{timm_model_name}".  Here are all the models {avail_pretrained_models}')
    raise Exception("Fix Network Name")

# Will download pretrained network if not already downloaded
feature_model = timm.create_model(timm_model_name, pretrained=True, num_classes=0)
feature_model.to(device)
# logit_model = timm.create_model(timm_model_name, pretrained=True)
# logit_model.to(device)


model_info = timm.data.resolve_data_config(args={},model=feature_model)

feature_model.eval()
# logit_model.eval()

def predict_feature(chip=None):
    # return dict({'logits':logit_model(chip), 'features':feature_model(chip)})
    return dict({'features':feature_model(chip)})
  
# x = torch.rand((1,3,model_info['input_size'][1], model_info['input_size'][2])).to(device)
# y = feature_model(x)
# feature_array = np.zeros((coco.n_annots,y.shape[1]))
# y = logit_model(x)
# logit_array = np.zeros((coco.n_annots,y.shape[1]))
# del y

# %% 
def prepare_image(x):
    x = torch.asarray(img).to(device)
    x = x.transpose(0,2)
    x = x / 255.0
    x[0] = x[0] - model_info['mean'][0] / model_info['std'][0]
    x[0] = x[0] - model_info['mean'][1] / model_info['std'][1]
    x[0] = x[0] - model_info['mean'][2] / model_info['std'][2]
    return x[None, :]


with Progress(
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
) as progress:
    task1 = progress.add_task("[red] Images...", total=coco.n_images)
    task2 = progress.add_task("[blue] Annots...", total=coco.n_annots)
    id = 0
    for gid in coco.images():
        coco_img = coco.coco_image(gid)
        annots = coco_img.annots()
        for ann in annots.objs:
            box = kwimage.Box.coerce(ann['bbox'], format='xywh')
            box = box.to_ltrb()
            delay_img = coco_img.imdelay()
            delay_img = delay_img.crop(box.quantize().to_slice(), clip=False, wrap=False)
            delay_img = delay_img.resize(model_info['input_size'][1:])
            img = delay_img.finalize()
            x = prepare_image(img)
            
            feats = predict_feature(x)
            ann['features'] = feats['features'].tolist()
            # ann['logits'] = feats['logits'].tolist()
            # plt.imshow(img)

            progress.update(task2, advance=1, refresh=True)
        
        progress.update(task1, advance=1, refresh=True)
        


# %%
coco._ensure_json_serializable()

coco.fpath = output_coco_filename
coco.dump()



# %%

# coco2 = coco.view_sql()
# coco2.pandas_table('images')


# %%

# annots = coco.annots()
# # %%


# data = annots.lookup('image_id')
# flags = [x%2==0 for x in data]
# annots.compress(flags)

# annots.__dict__

# %%
