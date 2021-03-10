Final Project

**Introduction
In the final project of the course we built an app for the scanner, similar to the app CamScanner.
The scanner app will assume that
	1. The document to be scanned is the main focus of the image.
	2. The document is rectangular, meaning it can be represented by four dots.

**Execution
	*Open the terminal from the folder where program Scanner.py is saved.
	*The program will be executed from the command line in the format:
	> python Scanner.py path_input_img path_output_img
		Where Scanner.py is the program name and path_input_img is a path to the image to be scanned.
		path_output is path to where the scanned image will be saved.
	*For example we opened as:
	>python Scanner.py C:\Users\Sarit\Desktop\Image_processing\Game.jpg C:\Users\Sarit\Desktop\Image_processing\Output.jpg
		
**Requirements 
	Before you continue, ensure you  have met the following requirements:
	*You have installed the latest version of Python 3.8
	*You are using a Windows machine.

*Packages:
	cycler==0.10.0
	decorator==4.4.2
	imageio==2.9.0  
	imutils==0.5.4  
	kiwisolver==1.3.1
	matplotlib==3.3.4
	networkx==2.5
	numpy==1.20.1
	opencv-python==4.5.1.48
	Pillow==8.1.0
	pyparsing==2.4.7
	python-dateutil==2.8.1
	PyWavelets==1.1.1
	scikit-image==0.18.1
	scipy==1.6.0
	six==1.15.0
	tifffile==2021.2.1
		
		
*Installation help:
	pip install opencv-python
	pip install matplotlib
	pip install imutils
	pip install scikit-image

**Program output
	Output.jpg – will be created in the current folder.
