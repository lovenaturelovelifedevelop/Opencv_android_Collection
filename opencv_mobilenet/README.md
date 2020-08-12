参考：
https://docs.opencv.org/3.4/d0/d6c/tutorial_dnn_android.html
参考：
https://blog.csdn.net/guyuealian/article/details/80570120

pb和pbtxt要匹配
tensorflow mobel模型：
 http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
匹配的pbtxt：
https://github.com/opencv/opencv_extra/blob/master/testdata/dnn/ssd_mobilenet_v2_coco_2018_03_29.pbtxt

Dnn.blobFromImage
第一个参数，InputArray image，表示输入的图像，可以是opencv的mat数据类型。
第二个参数，scalefactor，这个参数很重要的，如果训练时，是归一化到0-1之间，那么这个参数就应该为0.00390625f （1/256），否则为1.0
第三个参数，size，应该与训练时的输入图像尺寸保持一致。
第四个参数，mean，这个主要在caffe中用到，caffe中经常会用到训练数据的均值。tf中貌似没有用到均值文件。
第五个参数，swapRB，是否交换图像第1个通道和最后一个通道的顺序。
第六个参数，crop，如果为true，就是裁剪图像，如果为false，就是等比例放缩图像。
