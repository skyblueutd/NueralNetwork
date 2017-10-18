import pandas as pd
import logging

dropping_str = ["None", "none", ""]
# logger
def log():
	logging.basicConfig()
	logger = logging.getLogger('Preprocessing')
	logger.setLevel(logging.DEBUG)
	return logger

# read file into memory given path
def read(path):
	logger.debug("Received path: %s" % path)
	data = pd.read_table(path, sep=",", header=None)
	logger.debug("Successfully import dataset")
	return data

# remove none variable in data
def removeMissing(data):
	logger.debug("Started to remove missing value from dataset")
	data.dropna(); #drop any NaN data
	nrow, ncol = data.shape[0], data.shape[1]
	for i in range(ncol):
		#remove none and empty value
		data = data[~data[i].isin(dropping_str)]
	logger.debug("Removed missing value from dataset")
	return data

# if a variable is categorical, convert it to numeric value
# if a variable is numeric, normalize its value by substracting the mean, and dividied by the sd
def normalize(data):
	logger.debug("Started to normalization")
	nrow,ncol = data.shape[0],data.shape[1]
	for i in range(ncol):
		if(data[i].dtype == 'object'):
			data[i] = pd.factorize(data[i])[0]
		else:
			data_mean = data[i].mean()
			data_std = data[i].std()
			data[i] = data[i].apply(lambda x: (x-data_mean)/data_std)
	logger.debug("Data normalization finished")
	return data

# write dataset into disk given the output path
def write(data, path):
	logger.debug("Started to write clean data to %s" % path)
	data.to_csv(path, index=False)
	logger.debug("Finished writing clean data to %s" % path)

#ask input
inPath = input("Input path to dataset: ")
outPath = input("Output path: ")
logger = log()
#read data
data = read(inPath)
#remove missing data
data = removeMissing(data)
#normalize dataset
data = normalize(data)
#write dataset
write(data, outPath)