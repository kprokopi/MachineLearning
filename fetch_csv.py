#fetch csv files in zip from kaggle and unzip them
#It is assumed that the json file is already uploaded
#in folder named as .kaggle 

# connect to kaggle api and download files (zip)
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi({"username":"kprokopi","key":"9a28305a56476ed46a0ac149764d9610"})
api.authenticate()
files = api.competition_download_files("Instacart-Market-Basket-Analysis")

#unzip files
import zipfile
with zipfile.ZipFile('Instacart-Market-Basket-Analysis.zip', 'r') as zip_ref:
    zip_ref.extractall('./input')

import os
working_directory = os.getcwd()+'/input'
os.chdir(working_directory)
for file in os.listdir(working_directory):   # get the list of files
    if zipfile.is_zipfile(file): # if it is a zipfile, extract it
        with zipfile.ZipFile(file) as item: # treat the file as a zip
           item.extractall()  # extract it in the working directory
