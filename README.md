# Seez image vision on cars


# Data strategy
Use yolo classifications to grab images with cars. Join the labels of make/model/submodel on the image and perform image vision! Hurray!

Future: Use the yolo bbox as label along with car name to also output the bbox of the car.

# ML Strategy:

Use EfficientNet-B2/3 as it has excellent performance on benchmarks and not too huge..

We need to worry about being able to compress the model. Tensorflow should be best for this and easier to get on the phones, we can use TFLite.

I will probably be using a pretrained ImageNet model and freeze some of the parameters, since we have a lot of data it should be good to unfreeze many of them. Needs testing.


# Goals 

We're aiming for 80% top-1 accuracy and 95% top-5. 

 

# Budget

Hopefully this can be done cheaply using genesiscloud.com, got $50 from T-man

# Acquiring data

I am mainly using this query which grabs all the images with at least one car in them. I select the label(make,model, submodel) for each ad as well as p_hash names so they  can be used for downloading images from hetzner

```postgresql
with flatten as (
    select ad_id,
           p_hash,
           jsonb_array_elements(ad_images.yolo_classification) as yolo
    from app.ad_images
)

select ad_id, ma.name, mo.name, sm.name, hashish from app.ads
    join static.makes ma on ads.make_id = ma.id
    join static.models mo on ads.model_id = mo.id
    join static.submodels sm on ads.submodel_id = sm.id
    join static.models_submodels ms on mo.id = ms.model_id and ms.submodel_id = sm.id
    join
(select ad_id, string_agg(p_hash, ',') hashish
from flatten
where yolo ->> 'class_name' in ('car', 'truck', 'motorbike', 'bus')
group by ad_id) as cars
on cars.ad_id = ads.id
```

Save this as a CSV named `cars.csv` and run `python create_dataset.py` that should download all images and save them into `project_dir/data/<ad_id>/<p_hash>.jpg`. This large directory will work on modern file systems and is relatively easy to fetch batches from in TF.


# Acquiring pre-trained model

TF2 is new so resources are scarce, but EfficientNet-B2 can be downloaded here as of 16/03/20
https://github.com/tensorflow/tpu/tree/0337e3ba9285206cb4cb0562ffdb432f020846c9/models/official/efficientnet#2-using-pretrained-efficientnet-checkpoints 
Grab the noisystudent model

It can then be converted like so
```shell script
$ ./convert_efficientnet.sh --target_dir dist
```
