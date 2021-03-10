Assignment 3

**Introduction
	In this exercise, given a pair of images like the ones above, we will sew them to create a panoramic scene.
	It is important to note that the two images should share some common area.
	Also, the solution should work even if the images have differences in one or more of the following aspects:
	- Scaling
	- angle
	- Device from which the image was taken
**Execution
	*Open the terminal from the folder where program Panorama.py is saved.
	*The program will be executed from the command line in the format:
	> python Panorama.py path_left_img path_right_img path_output
		Where Panorama.py is the program name and path_left is a path to the left image and path_left is a path to the left image.
		path_output is path to where the panoramic image will be saved.
	*For example we opened as:
	>python Panorama.py C:\Users\Sarit\PycharmProjects\Panorama\left.jpg C:\Users\Sarit\PycharmProjects\Panorama\right.jpg C:\Users\Sarit\Pycharm
		Projects\Panorama\Panorama.jpg


**Requirements 
	Before you continue, ensure you  have met the following requirements:
	*You have installed the latest version of Python 3.8
	*You are using a Windows machine.
*Packages:
Pillow	8.1.0	8.1.0
astroid	2.4.2	2.4.2
certifi	2020.12.5	2020.12.5
chardet	4.0.0	4.0.0
colorama	0.4.4	0.4.4
cycler	0.10.0	0.10.0
idna	2.10	3.1
isort	5.7.0	5.7.0
kiwisolver	1.3.1	1.3.1
lazy-object-proxy	1.4.3	1.5.2
matplotlib	3.3.3	3.3.3
mccabe	0.6.1	0.6.1
numpy	1.19.5	1.19.5
opencv-contrib-python	4.5.1.48	4.5.1.48
opencv-python	4.5.1.48	4.5.1.48
pip	20.3.3	20.3.3
pylint	2.6.0	2.6.0
pyparsing	2.4.7	2.4.7
python-dateutil	2.8.1	2.8.1
python-resize-image	1.1.19	1.1.19
requests	2.25.1	2.25.1
resize-image	0.4.0	
setuptools	51.1.2	51.1.2
six	1.15.0	1.15.0
toml	0.10.2	0.10.2
urllib3	1.26.2	1.26.2
wrapt	1.12.1	1.12.1
		
		
*Installation help:
	pip install opencv-python
	pip install numpy==1.19.3 (numpy 1.19.14 has problems with windows when writing this document)
	pip install pillow
	pip install matplotlib
	pip install opencv-contrib-python


**Program output
	Print – Start time of the program, end time of the program and the total run time.
	Panorama.jpg – will be created in the current folder.
