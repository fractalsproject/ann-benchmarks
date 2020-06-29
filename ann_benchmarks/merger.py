import h5py
import numpy
import os
import random
import sys
from urllib.request import urlopen
from urllib.request import urlretrieve
from distance import dataset_transform

def write_output(train, test, fn, distance, point_type='float', count=100):
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
	import sklearn.model_selection
	print('Splitting %d*%d into train/test' % X.shape)
	return learn.model_selection.train_test_split(X, test_size=test_size, random_state=1)






#with open("base_05", "ab") as myfile, open("base_06", "rb") as file2:
#	myfile.write(file2.read())

raw_file = numpy.fromfile("base_06", dtype=numpy.float32)
true_vec = raw_file.reshape(-1, 1 + raw_file.view(numpy.int32)[0])[:, 1:]
print("Dim: " + str(len(true_vec[0])))
print("len: " + str(len(true_vec)))

fv =  true_vec
X_train, X_test = train_test_split(fv)
write_output(X_train, X_test, out_fn, 'angular')
