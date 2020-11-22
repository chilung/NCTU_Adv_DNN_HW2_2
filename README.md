## Environment preparation
conda install -c conda-forge argparse
conda install -c conda-forge tqdm
## Model conversion
In order to convert the weights from darknet to pytorch, you need to prepare some thing:
1. git clone this repository into your local environment.
2. download the test dataset from the link [here](https://drive.google.com/file/d/1nswVLQSGupsRympzb3tUv3L94d7ADPmi/view?usp=sharing)
3. unzip the dataset into the directory of ./NCTU_Adv_DNN_HW2_2 and you can see a new directory of test.
4. download the pretrained darknet weights. You can use this pretrained weights [here](https://drive.google.com/file/d/1lDDQ_JJmW0hv4SNGkcT1yiWcPxf459Us/view?usp=sharing)
5. put the pretrained weights, i.e., yolo-obj_mAP_93.6.weights in the directory of ./NCTU_Adv_DNN_HW2_2/exec_env.
6. perform the demo.py:

In Linux:

python demo.py -cfgfile='./exec_env/yolo-obj.cfg' -weightfile='./exec_env/yolo-obj_mAP_93.6.weights' -imgfiles='./exec_env/valid.txt' -namesfile='./exec_env/obj.names'

In Windows:

python demo.py -cfgfile='.\exec_env\yolo-obj.cfg' -weightfile='.\exec_env\yolo-obj_mAP_93.6.weights' -imgfiles='.\exec_env\valid.txt' -namesfile='.\exec_env\obj.names'
