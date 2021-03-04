Assignment 2 - OCR of Handwritten Hebrew
Author 1: Sarit Divekar
ID: 327373684
Author 2: Hadar Bar oz
ID: 204460737

**Introduction
	In this project you will use the k-Nearest Neighbor algorithm to classify images of letters from a repository HHD_0,
	which consists of handwritten letters. The HHD_v0 repository contains around 5000 images of letters Individuals. These images are divided into two groups (folders) (TEST and TRAIN, each of which is a group these are divided into 27 sub-folders (sub-folders), 0 being the first letter of the alphabet and 60 being the last. Each folder contains images of a particular letter from the Hebrew alphabet.
	*Details about the HHD_v0 database can be found at 
	I. Rabaev, B. Kurar Barakat, A. Churkin and J. El-Sana. The HHD Dataset. 
	The 17th International Conference on Frontiers in Handwriting Recognition, pp. 228-233, 2020.
	https://www.researchgate.net/publication/343880780_The_HHD_Dataset

**Execution
	*Open the terminal from the folder where program knn_classifier.py is saved.
	*The program will be executed from the command line in the format:
	> python knn_classifier.py path
		Where knn_classifier.py is the program name and path is a path to the folder with the repository(HHD_v0).
	*For example we opened as:
	>python knn_classifier.py C:\Users\Sarit\PycharmProjects\HW2\hhd_dataset

**Requirements 
	Before you continue, ensure you  have met the following requirements:
	*You have installed the latest version of Python 3.8
	*You are using a Windows machine.
*Packages:
Pillow	8.0.1	8.0.1
PyWavelets	1.1.1	1.1.1
cycler	0.10.0	0.10.0
decorator	4.4.2	4.4.2
imageio	2.9.0	2.9.0
joblib	0.17.0	0.17.0
kiwisolver	1.3.1	1.3.1
matplotlib	3.3.3	3.3.3
networkx	2.5	2.5
numpy	1.19.4	1.19.4
opencv-python	4.4.0.46	4.4.0.46
pandas	1.1.4	1.1.5
pip	20.3	20.3.1
pyparsing	2.4.7	2.4.7
python-dateutil	2.8.1	2.8.1
pytz	2020.4	2020.4
scikit-image	0.17.2	0.17.2
scikit-learn	0.23.2	0.23.2
scipy	1.5.4	1.5.4
setuptools	50.3.2	51.0.0
six	1.15.0	1.15.0
sklearn	0.0	0.0
threadpoolctl	2.1.0	2.1.0
tifffile	2020.11.26	2020.12.8
		
		
*Installation help:
	pip install opencv-python
	pip install numpy==1.19.3 (numpy 1.19.14 has problems with windows when writing this document)
	pip install pandas
	pip install pillow
	pip install sklearn
	pip install scikit-image

**Program output
	Print – Start time of the program, end time of the program and the total run time.
	result.txt – will be created in the current folder.
	confusion_matrix.csv – will be created in the current folder.
