# Seez image vision on cars

Imagevision on car pictures, find a car model from a picture! Currently WIP, demo can be found 

## Dataset
Massive set of car images scraped from mainly middle eastern craigslist-like clones, given that we know what car is being sold, we use yolov3 to tag images that actually feature the full car

Given the known car images, we can join the labels of make/model/submodel on the image and perform image vision on image, label examples. Hurray!

It should be noted that many listings are garbage and will introduce heavy label noise, alas we should be able to work against this, NN are very resilient towards label noise.

download_hetzner.py grabs the dataset over webdav. One needs a cars.csv file formatted:
```csv
ad_id, make, model, submodel, "image_id1, image_id2"
```
Note that one car has many pics here..

running download_hetzner.py  should download all images and save them into `project_dir/data/<ad_id>/<p_hash>.jpg`. This large directory will work on modern file systems and will even be trainable using pytorch/TF2 on SSDs.

However, it is incredibly wasteful using such  a format and scales better with tfrecords.

So given the above directory, we can create tfrecords (at this point  we associate labels as well). Run dataset.Dataset.write() to accomplish this

Outputs tfrecords to `project_dir/records/<train and val records>`



## ML Strategy:

Use EfficientNet-B2/3 as it has excellent performance on benchmarks and not too huge..

We need to worry about being able to compress the model. Tensorflow should be best for this and easier to get on the phones, we can use TFLite.

I will probably be using a pretrained ImageNet model and freeze some of the parameters, since we have a lot of data it should be good to unfreeze many of them. Needs testing.


Update: Using fully pretrained ImageNet model simply to kickstart the learning, there's no need to freeze anything, we want to maximize performance

## Goals 

We're aiming for 80% top-1 accuracy and 95% top-5. 

Future goal: Use the yolo bbox as label along with car name to also output the bbox of the car.
 



Save this as a CSV named `cars.csv` and run `python create_dataset.py` that should download all images and save them into `project_dir/data/<ad_id>/<p_hash>.jpg`. This large directory will work on modern file systems and is relatively easy to fetch batches from in TF.

