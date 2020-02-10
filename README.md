
# ReGNN
We setup our experiment on an Nvidia 1080Ti GPU and 256G memory on CentOs. 

Experiment Environment
-------
* python 3.6.5
* tensorflow-gpu 1.12
* numpy

Project Struct
------
> ### ReGNNdatasets/
* data_preprocess.py  ------ the data process file for three datasets.

* YoochooseSubDataset/
>> * all_train_seq.txt  ------ all train sequence dadaset
>> * test.txt  ------ test dataset
>> * train.txt  ------ train dataset

> ### ReGNN/program/
* main.py                   ------ this is the program entry
* ReGNN.py             ------ the major ReGNN model
* utils.py                  ------ some helper functions we used

Recommended Setup
------
You can run the main.py directly for easily running the program. 
If you run the code on linux, just running the following command:<br>
<br>
      `python main.py`
