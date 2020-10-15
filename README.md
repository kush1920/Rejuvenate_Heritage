# Rejuvenate_Heritage
Tech4Heritage Hackathon
* Team Name - Ancient_AI
* Team_ID - 128

## Problem Statement
* Create a Generator Model that can regenerate broken paintings with filling of missing parts (discontinuities):

	* Build a function that can objectively identify the discontinuities and the missing elements.
	* Build functions that can choose the right missing elements for the various types of missing elements identified in the above-mentioned functions. The choice of the right missing elements will be driven by the learning of the data sets in this problem statement.
	* Create multiple outputs out of the above-mentioned functions and create functions to choose the most relevant output (optimization of the functions).

* Create a Comparator or Discriminator Model to compare results of the outputs of the generative model with real image and provide probabilities of accuracy:
	* Create a function that can compare generated output with the accepted output and calculate level of accuracy ▪ Based on multiple data sets, calculate and provide probability functions mentioning the expected level of accuracy of restoration for any image that enters the generative model.
	* Train the generative model accordingly along with further training of this comparator model. Try to reach “equilibrium”.
* Objective at the end of Hackathon:
	* Create a system that reads a damaged image (with missing portions), identifies the areas that are damaged, and uses a model to fill the spaces that are missing using scientific assumptions drawn out of the sample datasets, effectively being able to restore any image similar to the sample dataset shared.

![Tech4Heritage](https://github.com/kush1920/Rejuvenate_Heritage/blob/main/Images/Tech4Heritage_Poster.jpg)

## SOLUTION APPROACH

* We required a solution that would first detect the spots/irregularities in the image and then try to fill the spots based on the surrounding region of the spots/irregularities and other images in the dataset on which our model is trained.
	* The first requirement was to build a dataset on our own , since datasets on damaged and corresponding clean paintings is not readily available. Steps for creating dataset -
		* We took some damaged pantings as shown in **data/Damaged/** folder . With Opencv we did color segmentation and took out spots from the damaged images . We then renamed the damaged images according to their lower and higher hsv values which we got from color segmentation.(This can be seen in **src/editing.ipynb**)
		* Then we took some clean paintings as shown in **data/Clean/** folder . We then overlayed the spots taken out from the damaged images as in above step on the clean images . (This can be seen in **src/generate_dataset.ipynb**)
		* Then to add some old affect and tint we applied Neural Style Transfer on the overlayed images . We downloaded some textures for style transfer to be applied on the paintings passed through the above step . The textures are in **data/Textures/** folder .(This can be seen in **src/generate_dataset.ipynb**)
		* Link for the dataset we created - ![Dataset](https://drive.google.com/drive/folders/1KgaaPV0NrPlJj-FVv-DR0P8cql4G0LAQ?usp=sharing)

* 

## Workflow
![Workflow ANPR](https://github.com/kush1920/Automatic-Number-Plate-Recognition/blob/master/Images%20and%20Videos/Images/workflow.jpg)

## Limitations

- Restrictions on Camera's Field Of View , Resolution an Frame Rate .
- A better OCR trained on strong GPU's will give better results .


## Future Improvement 

- Developments in ML and DL can give more precise and accurate outputs .
- Reduction in size of camera and microprocessor can save space .


## Team Members 
1. [Atharva Karanjgaokar](https://github.com/atharva2702)
2. [Kushagra Babbar](https://github.com/kush1920)

## Mentors
1. [Rishika Chandra](https://github.com/chandrarishika14)

## NOTE

DOWNLOAD YOLO WEIGHTS FROM [HERE](https://drive.google.com/file/d/1YZuTmP-c4b07z5mfhOAtP_V_oymP5_xG/view?usp=sharing)

## Refernces 

- Opencv - https://www.youtube.com/playlist?list=PLvVx8lH-gGeC8XmmrsG855usswhwt5Tr1

- Yolo object dtection - https://pjreddie.com/darknet/yolo/

- Pytessercat - https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/

- Object tracking - https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
