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


## Workflow
![Workflow ANPR](https://github.com/kush1920/Automatic-Number-Plate-Recognition/blob/master/Images%20and%20Videos/Images/workflow.jpg)

## Mechanical Aspects of Design

* Structure
<p align ="justify">
A box with 3 sliding doors, to hold the electronic components of the system. Easy to fabricate design, without the hassle of drilling, tapering and milling. Extruded cut in the front to fit the face of camera.
</p>

* DIMENSIONS:
BOX :  17 x  11  x  10  (cm)

* Model

![Model](https://github.com/atharva2702/Automatic-Number-Plate-Recognition/blob/master/Mechanical%20Design/CAD2.png)

## Electric Aspects of Design 

* Microcontroller
Raspberry Pi 3B+ 

* Camera
LOGITECH WEBCAM 

* Battery
12 volt battery powering the pi with Micro USB 2.0 outlet

## Cost Structure

| Components       | Cost(INR.)     |
| ---------------- |:--------------:|
| Raspberry Pi     | 3830.00        |
| Battery          | 1500.00        |
| LOGITECH WEBCAM  | 3500.00        | 
| Total            | 8830.00        |


## Applications

- Shopping Malls : To keep a record of vehicles entering
- Toll Plaza : On highways for security reasons 
- Parking Plazas : To detect and generate tax for parked vehicles automatically
- Educational or Government Institutions : To ensure only authorized vehicles are permitted inside 


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
