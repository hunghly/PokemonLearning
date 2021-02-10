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

ap = argparse.ArgumentParser(description="You must provide a query and output for Bing queries")
ap.add_argument("-q", "--query", required=True,
	help="search query to search Bing Image API for")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory of images")
args = vars(ap.parse_args())

# set your Microsoft Cognitive Services API key along with (1) the
# maximum number of results for a given search and (2) the group size
# for results (maximum of 50 per request)
MAX_RESULTS = 250
GROUP_SIZE = 50
# set the endpoint API URL
# URL = "https://api.cognitive.microsoft.com/bing/v7.0/images/search"
URL = "https://api.bing.microsoft.com/v7.0/images/search"

# when attempting to download images from the web both the Python
# programming language and the requests library have a number of
# exceptions that can be thrown so let's build a list of them now
# so we can filter on them
EXCEPTIONS = set([IOError, FileNotFoundError,
	exceptions.RequestException, exceptions.HTTPError,
	exceptions.ConnectionError, exceptions.Timeout])

# store the search term in a convenience variable then set the
# headers and search parameters
term = args["query"]
headers = {"Ocp-Apim-Subscription-Key" : keys.API_KEY}
params = {"q": term, "offset": 0, "count": GROUP_SIZE}

# initialize the total number of images downloaded thus far
total = 0

# loop over the estimated number of results in `GROUP_SIZE` groups
for offset in range(0, MAX_RESULTS, GROUP_SIZE):
    # loop over the results
    # print("results", results["images"])
    print("[INFO] making request for group {}-{} of {}...".format(offset, offset + GROUP_SIZE, MAX_RESULTS))
    params["offset"] = offset
    search = requests.get(URL, headers=headers, params=params)
    search.raise_for_status()
    results = search.json()
    print("[INFO] saving images for group {}-{} of {}...".format(offset, offset + GROUP_SIZE, MAX_RESULTS))
    for v in results["value"]:
        # try to download the image
        try:
            # make a request to download the image
            print("[INFO] fetching: {}".format(v["contentUrl"]))
            r = requests.get(v["contentUrl"], timeout=30)
            # build the path to the output image
            ext = v["contentUrl"][v["contentUrl"].rfind("."):]
            p = os.path.sep.join([args["output"], "{}{}".format(
                str(total).zfill(8), ext)])
            # write the image to disk
            f = open(p, "wb")
            f.write(r.content)
            f.close()
        # catch any errors that would not unable us to download the
        # image
        except Exception as e:
            # check to see if our exception is in our list of
            # exceptions to check for
            if type(e) in EXCEPTIONS:
                print("[INFO] skipping: {}".format(v["contentUrl"]))
                continue


        # # try to load the image from disk
        # image = cv2.imread(p)
        # # if the image is `None` then we could not properly load the
        # # image from disk (so it should be ignored)
        # if image is None:
        #     print("[INFO] deleting: {}".format(p))
        #     os.remove(p)
        #     continue
        # # update the counter
        # total += 1
