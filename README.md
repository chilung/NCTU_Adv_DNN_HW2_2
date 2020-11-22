In order to convert the weights from darknet to pytorch, you need to prepare some thing:
1. git clone this repository into your local environment.
2. download the test dataset from the link [here](https://drive.google.com/file/d/1nswVLQSGupsRympzb3tUv3L94d7ADPmi/view?usp=sharing)
3. unzip the dataset into the directory of ./NCTU_Adv_DNN_HW2_2/test
4. download the pretrained darknet weights. You can use this pretrained weights [here](https://drive.google.com/file/d/1lDDQ_JJmW0hv4SNGkcT1yiWcPxf459Us/view?usp=sharing)
5. perform the demo.py:
python demo.py -cfgfile='./exec_env/yolo-obj.cfg' -weightfile='./exec_env/yolo-obj-mAP_93.3.weights' -imgfiles='./exec_env/valid.txt' -namesfile='./exec_env/obj.names'
