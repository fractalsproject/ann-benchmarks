import h5py
import numpy
import os
import random
import sys
from urllib.request import urlopen
from urllib.request import urlretrieve
from distance import dataset_transform

def write_output(train, test, fn, distance, point_type='float', count=100):
	print("Starting ground truth to h5df")
	from ann_benchmarks.algorithms.bruteforce import BruteForceBLAS
	n = 0
	f = h5py.File(fn, 'w')
	f.attrs['distance'] = distance
	f.attrs['point_type'] = point_type
	print('train size: %9d * %4d' % train.shape)
	print('test size:  %9d * %4d' % test.shape)
	f.create_dataset('train', (len(train), len(train[0])), dtype=train.dtype)[:] = train
	f.create_dataset('test', (len(test), len(test[0])), dtype=test.dtype)[:] = test

	neighbors = f.create_dataset('neighbors', (len(test), count), dtype='i')
	distances = f.create_dataset('distances', (len(test), count), dtype='f')

	bf = BruteForceBLAS(distance, precision=train.dtype)
	train = dataset_transform[distance](train)
	test = dataset_transform[distance](test)
	bf.fit(train)
	queries = []
	for i, x in enumerate(test):
		if i % 1000 == 0:
			print('%d/%d...' % (i, len(test)))

		res = list(bf.query_with_distances(x, count))
		res.sort(key=lambda t: t[-1])
		neighbors[i] = [j for j, _ in res]
		distances[i] = [d for _, d in res]
	f.close()

def train_test_split(X, test_size=10000):
	print("Starting train test split")
	import sklearn.model_selection
	print('Splitting %d*%d into train/test' % X.shape)
	return learn.model_selection.train_test_split(X, test_size=test_size, random_state=1)


def trim_end(location, vector):return vector[:len(vector)-location]

def vector_order_valid(fv):
	count = 0
	print("Begining checks ...")
	for i in range(0,len(fv),97):
		if fv[i] != 96:
			count = count + 1
			print("no 96 at position:  "+ str(i) +" instead it is " + str(fv[i]))
	print("vector order check done, there were " + str(count) + "mismatches")
	print()
	print()
	if len(fv)/97 != 0:
		print("Vector has been clipped checking for clip location")
		for i in range(1,100):
			if fv[-i] == 96:
				print("final vector is only: " + str(i) + " dimentions long")
				fv_trim = trim_end(i,fv)
				break
	print("Done!")
	return fv_trim

def merge_base(out_fn):
	with open("base_00", "ab") as myfile, open("base_01", "rb") as file2:
	myfile.write(file2.read())
	fv = numpy.fromfile("base_00",dtype="int32")
	dim = fv.view(numpy.int32)[0]
	new = vector_order_valid(fv)
	new = new.reshape(-1, dim + 1)[:,1:]
	X_train, X_test = train_test_split(new)
	#write_output(X_train, X_test, out_fn, 'angular')

