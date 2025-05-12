# How to run
To run the program, first make sure you are using python 3.11. All our results were achieved on that python version as later versions have deprecated some of the functionality we are using. 

Please refer to requirements.txt for a comprehensive list of all modules used in our virtual environment. 

After downloading all the modules, please download the dataset using the this link: https://data.4tu.nl/file/99b5c721-280b-450b-b058-b2900b69a90f/410e41c7-dde5-413f-8722-3e112363a1a2 and unzip the folder in the project directory using its default name: chessred2k.

After that, you can use python main.py to run the project.

If you want to preprocess the images yourself, it is expected to take 3-6 hours, depending on your machine. The included folder dataset_cached will be used instead to speed up the process. It contains all detected chess boards.