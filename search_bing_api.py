from requests import exceptions
import argparse
import requests
import cv2
import os
import keys

# construct the argument parser and parse the arguments
"""
--query:  The image search query you’re using, which could be anything such as “pikachu”, “santa” or “jurassic park”.
--output:  The output directory for your images. My personal preference (for the sake of organization and sanity) is to separate your images into separate class subdirectories, so be sure to specify the correct folder that you’d like your images to go into (shown below in the “Downloading images for training a deep neural network” section).
"""

# ap = argparse.ArgumentParser(description="You must provide a query and output for Bing queries")
# ap.add_argument("-q", "--query", required=True,
# 	help="search query to search Bing Image API for")
# ap.add_argument("-o", "--output", required=True,
# 	help="path to output directory of images")
# args = vars(ap.parse_args())

print(keys.API_KEY)
