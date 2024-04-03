<h2>Tensorflow-Image-Segmentation-Cervical-Cancer (2024/04/04)</h2>

This is an experimental Image Segmentation project for Cervical-Cancer based on
the <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, and
<a href="https://drive.google.com/file/d/1DxqbpP6kxzBSdH9jA0gCp8o_znI6Pd-7/view?usp=sharing">
Cervical-Cancer-ImageMask-Dataset-V2.zip</a> 
<br>

<br>
Segmentation for test images of 2048x1536 size by <a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> Model<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/asset/segmentation_samples.png" width="720" height="auto">
<br>
<br>
Please see also our first experiment <a href="https://github.com/atlan-antillia/Image-Segmentation-Cervical-Cancer">Image-Segmentation-Cervical-Cancer</a>.
<br>
<br>
In this experiment, we have used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Cervical-Cancer Segmentation.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<br>

<h3>1. Dataset Citation</h3>
The original dataset used here has been taken from the following kaggle website:<br>
<b>Cervical Cancer largest dataset (SipakMed)</b><br>
<pre>
https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed
</pre>
<pre>
About Dataset
Please don't forget to upvote if you find this useful.
Context
Cervical cancer is the fourth most common cancer among women in the world, estimated more than 0.53 million 
women are diagnosed in every year but more than 0.28 million women’s lives are taken by cervical cancer 
in every years . Detection of the cervical cancer cell has played a very important role in clinical practice.

Content
The SIPaKMeD Database consists of 4049 images of isolated cells that have been manually cropped from 966 cluster
 cell images of Pap smear slides. These images were acquired through a CCD camera adapted to an optical microscope. 
 The cell images are divided into five categories containing normal, abnormal and benign cells.

Acknowledgements
IEEE International Conference on Image Processing (ICIP) 2018, Athens, Greece, 7-10 October 2018.

Inspiration
CERVICAL Cancer is an increasing health problem and an important cause of mortality in women worldwide. 
Cervical cancer is a cancer is grow in the tissue of the cervix . It is due to the abnormal growth of cell that 
are spread to the other part of the body.
Automatic detection technique are used for cervical abnormality to detect Precancerous cell or cancerous cell 
than no pathologist are required for manually detection process.
</pre>

<br>

<h3>
<a id="2">
2 Cervical-Cancer ImageMask Dataset
</a>
</h3>
 If you would like to train this Cervical-Cancer Segmentation model by yourself,
 please download the latest normalized dataset from the google drive 
<a href="https://drive.google.com/file/d/1DxqbpP6kxzBSdH9jA0gCp8o_znI6Pd-7/view?usp=sharing">
Cervical-Cancer-ImageMask-Dataset-V2.zip</a>,
Please refer to the dataset augmentation tool 
<a href="https://github.com/sarah-antillia/ImageMask-Datasset-Cervical-Cancer">ImageMask-Datasset-Cervical-Cancer</a>.

<br>

<br>
Please expand the downloaded ImageMaskDataset and place them under <b>./dataset</b> folder to be
<pre>
./dataset
└─Cervical-Cancer
    ├─test
    │  ├─Dyskeratotic
    │  │  ├─images
    │  │  └─masks
    │  ├─Koilocytotic
    │  │  ├─images
    │  │  └─masks
    │  ├─Metaplastic
    │  │  ├─images
    │  │  └─masks
    │  ├─Parabasal
    │  │  ├─images
    │  │  └─masks
    │  └─Superficial-Intermediate
    │      ├─images
    │      └─masks
    ├─train
    │  ├─Dyskeratotic
    │  │  ├─images
    │  │  └─masks
    │  ├─Koilocytotic
    │  │  ├─images
    │  │  └─masks
    │  ├─Metaplastic
    │  │  ├─images
    │  │  └─masks
    │  ├─Parabasal
    │  │  ├─images
    │  │  └─masks
    │  └─Superficial-Intermediate
    │      ├─images
    │      └─masks
    └─valid
        ├─Dyskeratotic
        │  ├─images
        │  └─masks
        ├─Koilocytotic
        │  ├─images
        │  └─masks
        ├─Metaplastic
        │  ├─images
        │  └─masks
        ├─Parabasal
        │  ├─images
        │  └─masks
        └─Superficial-Intermediate
            ├─images
            └─masks

</pre>
Please note that the pixel size of images and masks files in test dataset is 2048x1536 of the original SipakMed dataset, while  
the train and valid datasets created by the augmentation tool is 512x512.<br>
<br> 
In this experiment,for simplicity, we deal with <b>Metaplastic</b> category only.<br>
<br>
<b>Cervical-Cancer Metaplastic Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/asset/Metaplastic_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid dataset is not necessarily large. 
<br>

<br>
<b>Train_Metaplastic_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/asset/train_Metaplastic_images_sample.png" width="1024" height="auto">
<br>
<b>Train_Metaplastic_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/asset/train_Metaplastic_masks_sample.png" width="1024" height="auto">
<br>

<h2>
4 Train TensorflowUNet Model
</h2>
 We have trained Cervical-Cancer TensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/Cervical-Cancer and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>

<pre>
; train_eval_infer.config
; 2024/04/03 (C) antillia.com

[model]
model         = "TensorflowUNet"
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = False
num_classes    = 1
base_filters   = 16
base_kernels   = (5,5)
num_layers     = 7
dropout_rate   = 0.08
learning_rate  = 0.0001
clipvalue      = 0.5
dilation       = (2,2)
;loss           = "bce_iou_loss"
loss           = "bce_dice_loss"
metrics        = ["binary_accuracy"]
show_summary   = False

[train]
epochs        = 100
batch_size    = 4
steps_per_epoch  = 200
validation_steps = 100
patience      = 10
;metrics       = ["iou_coef", "val_iou_coef"]
metrics       = ["binary_accuracy", "val_binary_accuracy"]
model_dir     = "./models"
eval_dir      = "./eval"
image_datapath = "../../../dataset/Cervical-Cancer/train/Metaplastic/images/"
mask_datapath  = "../../../dataset/Cervical-Cancer/train/Metaplastic/masks/"
create_backup  = False
learning_rate_reducer = True
reducer_patience      = 5
save_weights_only = True

[eval]
image_datapath = "../../../dataset/Cervical-Cancer/valid/Metaplastic/images/"
mask_datapath  = "../../../dataset/Cervical-Cancer/valid/Metaplastic/masks/"

[test] 
image_datapath = "../../../dataset/Cervical-Cancer/test/Metaplastic/images/"
mask_datapath  = "../../../dataset/Cervical-Cancer/test/Metaplastic/masks/"

[infer] 
images_dir    = "./mini_test/Metaplastic/images"
output_dir    = "./mini_test_output"
merged_dir    = "./mini_test_output_merged"
binarize      = True

[segmentation]
colorize      = True
black         = "black"
white         = "green"
blursize      = None

[mask]
blur      = True
blur_size = (3,3)
binarize  = False
;threshold = 128

</pre>
The training process has just been stopped at epoch 55 by an early-stopping callback as shown below.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/asset/train_console_output_at_epoch_55.png" width="720" height="auto"><br>
<br>
<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
3.2 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Cervical-Cancer.<br>
<pre>
./2.evaluate.bat
</pre>
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/asset/evaluate_console_output_at_epoch_55.png" width="720" height="auto">
<br><br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/evaluation.csv">evaluation.csv</a><br>
The loss score for this test dataset is not so low as shown below.<br>
<pre>
loss,0.2506
binary_accuracy,0.9758
</pre>

<h2>
3.3 Inference
</h2>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Cervical-Cancer.<br>
<pre>
./3.infer.bat
</pre>
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
Metaplastic_mini_test_images<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/asset/Metaplastic_test_images.png" width="1024" height="auto"><br>
Metaplastic_mini_test_mask(ground_truth)<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/asset/Metaplastic_test_masks.png" width="1024" height="auto"><br>

<br>
Inferred test masks<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
Merged test images and inferred masks<br> 
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/asset/mini_test_output_merged.png" width="1024" height="auto"><br> 


Enlarged samples<br>
<table>
<tr>
<td>
test/image Metaplastic_244.jpg<br>
<img src="./dataset/Cervical-Cancer/test/Metaplastic/images/Metaplastic_244.jpg" width="512" height="auto">

</td>
<td>
Inferred merged Metaplastic_244.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/mini_test_output_merged/Metaplastic_244.jpg" width="512" height="auto">
</td> 
</tr>


<tr>
<td>
test/image Metaplastic_245.jpg<br>
<img src="./dataset/Cervical-Cancer/test/Metaplastic/images/Metaplastic_245.jpg" width="512" height="auto">

</td>
<td>
Inferred merged Metaplastic_245.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/mini_test_output_merged/Metaplastic_245.jpg" width="512" height="auto">
</td> 
</tr>



<tr>
<td>
test/image Metaplastic_247.jpg<br>
<img src="./dataset/Cervical-Cancer/test/Metaplastic/images/Metaplastic_247.jpg" width="512" height="auto">

</td>
<td>
Inferred merged Metaplastic_247.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/mini_test_output_merged/Metaplastic_247.jpg" width="512" height="auto">
</td> 
</tr>

<tr>
<td>
test/image Metaplastic_248.jpg<br>
<img src="./dataset/Cervical-Cancer/test/Metaplastic/images/Metaplastic_248.jpg" width="512" height="auto">

</td>
<td>
Inferred merged Metaplastic_248.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/mini_test_output_merged/Metaplastic_248.jpg" width="512" height="auto">
</td> 
</tr>


<tr>
<td>
test/image Metaplastic_253.jpg<br>
<img src="./dataset/Cervical-Cancer/test/Metaplastic/images/Metaplastic_253.jpg" width="512" height="auto">

</td>
<td>
Inferred merged Metaplastic_253.jpg<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Cervical-Cancer/mini_test_output_merged/Metaplastic_253.jpg" width="512" height="auto">
</td> 
</tr>


</table>
<br>

<h3>
References
</h3>
<b>1. Cervical Cancer largest dataset (SipakMed)</b><br>
<pre>
https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed
</pre>
<pre>
About Dataset
Please don't forget to upvote if you find this useful.
Context
Cervical cancer is the fourth most common cancer among women in the world, estimated more than 0.53 million 
women are diagnosed in every year but more than 0.28 million women’s lives are taken by cervical cancer 
in every years . Detection of the cervical cancer cell has played a very important role in clinical practice.

Content
The SIPaKMeD Database consists of 4049 images of isolated cells that have been manually cropped from 966 cluster
 cell images of Pap smear slides. These images were acquired through a CCD camera adapted to an optical microscope. 
 The cell images are divided into five categories containing normal, abnormal and benign cells.

Acknowledgements
IEEE International Conference on Image Processing (ICIP) 2018, Athens, Greece, 7-10 October 2018.

Inspiration
CERVICAL Cancer is an increasing health problem and an important cause of mortality in women worldwide. 
Cervical cancer is a cancer is grow in the tissue of the cervix . It is due to the abnormal growth of cell that 
are spread to the other part of the body.
Automatic detection technique are used for cervical abnormality to detect Precancerous cell or cancerous cell 
than no pathologist are required for manually detection process.
</pre>

<b>2. EfficientNet-Cervical-Cancer</b><br>
Toshiyuki Arai @antillia.com<br>
<pre>
https://github.com/atlan-antillia/EfficientNet-Cervical-Cancer
</pre>

<b>3. Liquid based cytology pap smear images for multi-class diagnosis of cervical cancer</b><br>
<pre>
https://data.mendeley.com/datasets/zddtpgzv63/4
</pre>

<b>4. Pap-smear Benchmark Data For Pattern Classiﬁcation<br></b>
Jan Jantzen, Jonas Norup , George Dounias , Beth Bjerregaard<br>

<pre>
https://www.researchgate.net/publication/265873515_Pap-smear_Benchmark_Data_For_Pattern_Classification
</pre>
<b>5. Deep Convolution Neural Network for Malignancy Detection and Classification in Microscopic Uterine Cervix Cell Images</b><br>
Shanthi P B,1 Faraz Faruqi, Hareesha K S, and Ranjini Kudva<br>
<pre>
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7062987/
</pre>

<b>6. DeepCyto: a hybrid framework for cervical cancer classification by using deep feature fusion of cytology images</b><br>
Swati Shinde, Madhura Kalbhor, Pankaj Wajire<br>
<pre>
https://www.aimspress.com/article/doi/10.3934/mbe.2022301?viewType=HTML#b40
</pre>


