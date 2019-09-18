Welcome to the VLML documentation!
==================================

Machine learning submodule in Video Library project handles the problem of Automatic Video Tag Suggestion and Content Categorization.
It uses the network architecture of YOLOv3 (You Only Look Once) object detector (`paper <https://pjreddie.com/media/files/papers/YOLOv3.pdf>`_)
fine-tuned on the `VEDAI <https://downloads.greyc.fr/vedai/>`_. dataset, and pre-trained on `Google Open Images v5 dataset <https://storage.googleapis.com/openimages/web/index.html>`_.


Installation process
====================

Requirements if running on local machine:
	* Python 3.6 or higher and Anaconda installed

	* CUDA (tested on CUDA 10.1) `Download it here <https://developer.nvidia.com/cuda-downloads>`_

	* CDNN (note that CUDNN must match the CUDA version) `Download it here <https://developer.nvidia.com/cudnn>`_

	* PyTorch 0.4 or higher (Please note that using PyTorch 0.3 will effect the detector) 
		If using Anaconda you can run
		``conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`` (change to your CUDA Toolkit version)

	* Necessary libraries (one of the following options):
		1. Custom Anaconda yolo environment (*yolo.yml*)
			* Download the yolo.yml file from `here <https://drive.google.com/open?id=131TYV34-pQv7SrjvX8-C7syYHIHqWr-d>`_
			* Run ``conda env create -f yolo.yml``

		2. Installing the libraries from PyPi (*requirements.txt*)
			* Download the requirements.txt file from `here <https://drive.google.com/open?id=1JJ2RDYWuYv_pmK8wah2-pNwuBxgOTZ-H>`_
			* Run ``pip install requirements.txt``

		Please note that some of the libraries may not have been installed through requirements.
		If openCV is not installed, please run ``pip install opencv-python``
	

Requirements for running in custom Docker container:
	* Get CUDA running on Docker (nvidia-smi)
	* Docker file (DockerFileML)


Weights files can be found here `YOLO on COCO <https://drive.google.com/open?id=1A52PSNSCN2hgsjaT8zGxujgwvLRMpQZ6>`_ and here `YOLO on Custom Dataset <https://drive.google.com/open?id=1PIYyZcLblizLjbNbXb2oUgDoq-lboo01>`_

Run detector
============

The object detector is integrated into the Video Library project, but can run individualy on any video input.

Before running the detector, you can check if CUDA is active by running:
::
   import os
   os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
   torch.cuda.is_available()

When not using the whole video library system, one can run the detection from command line, running: ``python detect_cmd.py`` with optional parameters ``--videoid (identifying hash of the video)`` ``--video (path to video file)`` ``--batch_size (batch size)`` ``--confidence (confidence)`` ``--nms_thresh (threshold)``

Modifying the code
==================

Changing the detect file
------------------------

The main function is located in detect.py file. It is based on the Darknet Implementation `here <(https://pjreddie.com/darknet/yolo/)>`_
*Important note*: By default, Darknet changes the resolution of input images to the size of 416 x 416. You can change this in *darknet.py* by changing this line of code 
:: 
	img = cv2.resize(img, (416,416)
but simultaneously changing resolution in *detect.py* 
:: 
	reso = 416 
Again,  whatever value you chose, rememeber it should be a **multiple of 32 and greater than 32**.

*On different scales*: YOLO v3 makes detections across different scales, each of which deputise in detecting objects of different sizes depending upon whether they capture coarse features, fine grained features or something between. You can experiment with these scales changing this line
:: 
	scales = 1,2,3

There are three **important files** that are not to be missed: weights file (*yolo-openimages.weights*), cfg file (*cfg/yolov3-openimages.cfg*) and names file (*data/openimages.names*).

If needed, you can update these files by chaning following lines of code 
::
   weights = osp.join(path,'yolov3-openimages.weights')
   cfgfile = osp.join(path,'cfg','yolo3-openimages.cfg')
   classes = load_classes(osp.join(path, 'data', 'openimages.names'))
Please note that any changes of the following files could change the input tensor size and lead to an error, so please change cfg and weights accordingly if training on the custom dataset, and change classes in the names file.

Changing the darknet file
-------------------------

Darknet file consists of the following parts:
::
	def parse_cfg(cfgfile):
Gets the cfg file and stores each block as a dict. The attributes of the blocks and their values are stored as key-value pairs in the dictionary.
::
	class MaxPoolStride1(nn.Module):
Controls how the filter convolves around the input volume
::
	class Upsample(nn.Module):
Controls how the filter convolves around the input volume
::
	class EmptyLayer(nn.Module):
*Important note:* When using torch.cat we put a dummy layer in the place of a proposed route layer, and then perform the concatenation directly in the forward function of the nn.Module object representing darknet. **Please do not remove this layer**.
::
	class Darknet(nn.Module):
Performs operation on the layers in the YOLO architecture.

It is highly recommended not to change the Darknet function, since it behaves only as an additional wrapper for implementing behavior of the layers.
Note that if there is no CUDA available, we don't convert the parameter to GPU.Instead. we copy the parameter and then convert it to CPU

Training on custom dataset
==========================

Training from scratch on the custom dataset
-------------------------------------------
1. *Important*: Please make sure your data is in Darknet format
::
	..dataset\images\val\image1.jpg  # image
	..dataset\labels\val\image1.txt  # label	

Please follow these rules:
	* One row per object.
	* Each row is class x_center y_center width height format.
	* Box coordinates must be in normalized xywh format (from 0 - 1).
	* Classes are zero-indexed (start from 0).

2. Separate data in train and test set.

3. Replace ``.names`` file and ``.data`` file to fit your new classes.

4. Each YOLO layer has 255 outputs: 85 outputs per anchor [4 box coordinates + 1 object confidence + 80 class confidences], times 3 anchors. If you use fewer classes, you can reduce this to ``[4 + 1 + n] * 3 = 15 + 3*n`` outputs, where n is your class count. 
This modification should be made to the output filter preceding each of the 3 YOLO layers. Also modify classes=80 to classes=n in each YOLO layer, where n is your class count.

5. Run darknet file normally.

Fine-tuning pre-trained model on custom dataset
-----------------------------------------------
Note that fine-tuning is performed via top-down domain adaptation, where the weights are adapted to the new custom dataset distribution.

1. Prepare the dataset like described in section Training from scratch.

2. Set flag ``random=1`` in your .cfg file - it will increase precision by training Yolo for different resolutions.

3. Set learning rate to be drastically lower in the .cfg file ``learning-rate=0.000001``.

4. In the .cfg file, find the first appearance of ``######################``. 
Look at the shortcut layer above, and insert ``stopbackward = 1`` (it is usually line 548).

5. Run darknet file normally.

My experiment with fine-tuning on VEDAI dataset
-----------------------------------------------

I have modified the VEDAI dataset and prepared it for the YOLOv3. 
	* Annotation files are available for `Windows <https://drive.google.com/open?id=1EB_w6PUSEb_Sgh0FN9MoNVB9HUhh9Sti>`_, and for Linux on the official website.
	* You can find more information about the dataset itself `here <https://downloads.greyc.fr/vedai/>`_. 
I have later performed fine-tuning like described, with using only the fraction of the data for the rest of the classes (Validation set from open images, can be downloaded `here <https://storage.googleapis.com/openimages/web/index.html>`_).

Indices and search
==================

* :ref:`genindex`
* :ref:`search`
