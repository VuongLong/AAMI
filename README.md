Interpolation missing join base on PCA:

	- open Terminal in your OS
	
	- go to your wanted folder and clone project:
	
	- git clone https://github.com/VuongLong/AAMI/blob/master/main.py

Data and parameter before runing:
	
	data.py: this file will show what data will be used, change reference data, testing data, ... in this file.
		
		- TEST_DIR = ["./fastsong/fastsong7.txt"] TEST_DIR is the location of original file
		
		- SELECTED_TEST_DATA = [[450, 550]] SELECTED_TEST_DATA is test patch of original file
		
		- MISSING_MAP_DIR = "./fastsong/fastsong7/12/16.txt" MISSING_MAP_DIR is location of test file
	
	main_DrYu_final: this file contains adaboost algorithm, change the threshold parameter, power coefficent, ... in this file
		
		- self.threshold = 0.7
		
		- self.power_coefficient = 3
	
	algorithm: contains computing alpha function, svd, normalization, that correspond to our T F formular

run:

	- cd Dr_YU
	
	- python main_DrYu_final.py


