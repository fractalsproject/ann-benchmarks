import h5py
import numpy
import os
import random
import sys
from urllib.request import urlopen
from urllib.request import urlretrieve
from ann_benchmarks.distance import dataset_transform
import time
from ann_benchmarks.algorithms.bruteforce import BruteForceBLAS

def write_output(train, test, fn, distance, point_type='float', count=10):
    print("Starting ground truth to h5df")
    from ann_benchmarks.algorithms.bruteforce import BruteForceBLAS
    n = 0
    f = h5py.File(fn, 'w')
    f.attrs['distance'] = distance
    f.attrs['point_type'] = point_type
    print('train size: %9d * %4d' % train.shape)
    print('test size:  %9d * %4d' % test.shape)
    
    #MEM error about half the time
    try:
        f.create_dataset('train', (len(train), 96), chunks = True, dtype=test.dtype)[:] = train
        f.create_dataset('test', (len(test), len(test[0])), dtype=test.dtype)[:] = test
        neighbors = f.create_dataset('neighbors', (len(test), count), dtype='i')
        distances = f.create_dataset('distances', (len(test), count), dtype='f')
        bf = BruteForceBLAS(distance)
        train = dataset_transform[distance](train)
        test = dataset_transform[distance](test)
        bf.fit(train)
        queries = []
        for i, x in enumerate(test):
            if i % 10 == 0:
                print('%d/%d...' % (i, len(test)))

            res = list(bf.query_with_distances(x, count))
            res.sort(key=lambda t: t[-1])
            neighbors[i] = [j for j, _ in res]
            distances[i] = [d for _, d in res]
        f.close()
    except Exception as e:
        print("FAILED because of")
        print(e)
        f.close()

def trim_end(location, vector):return vector[:len(vector)-location]

def train_test_split(X, test_size=10000):
	print("Starting train test split")
	import sklearn.model_selection
	print('Splitting %d*%d into train/test' % X.shape)
	return sklearn.model_selection.train_test_split(X, test_size=test_size, random_state=1)


def vector_order_valid(fv):
	count = 0
	print("Begining checks ...")
	for i in range(0,len(fv),97):
		if fv[i] != 96:
			count = count + 1
			print("no 96 at position:  "+ str(i) +" instead it is " + str(fv[i]))
	print("vector order check done, there were " + str(count) + " mismatches")
	print()
	print()
	if len(fv)/97 != 0:
		print("Vector has been clipped checking for clip location")
		for i in range(1,100):
			if fv[-i] == 96:
				print("final vector is only: " + str(i) + " dimentions long")
				#trim_sz = os.stat("base_00").st_size - 4*i
				#f = open("base_00","wb")
				#f.truncate(trim_sz)
				#f.close()

				fv_new = trim_end(i,fv)
				break
	print("Done!")
	return fv_new

if __name__== "__main__":

    if 0:
        start = time.time()
        print("Starting Merge.....")
        print("Merging 0 and 1...")
        with open("merge_0123", "ab") as myfile, open("/home/braden/GNOIMI/base_01", "rb") as file1:
            myfile.write(file1.read())
            myfile.flush()
            myfile.close()
        print("Merging 0 and 1 and 2...")
        with open("merge_0123", "ab") as myfile, open("/home/braden/GNOIMI/base_02", "rb") as file2:
            myfile.write(file2.read())
            myfile.flush()
            myfile.close()
        print("Merging 0 and 1 and 2 and 3...")
        with open("merge_0123", "ab") as myfile, open("/home/braden/GNOIMI/base_03", "rb") as file3:
            myfile.write(file3.read())
            myfile.flush()
            myfile.close()
        print("Merge Done")
        end = time.time()
        print("Merge done at " +str(end-start) + " seconds")

    print("Creating HD5PY DB...")
    print("Getting FV from merge...")
    fv = numpy.fromfile("merge_0123",dtype="int32")
    dim = fv.view(numpy.int32)[0]
    print("Validating...")
    fv_new = vector_order_valid(fv)
    print("Reshaping...")
    new = fv_new.reshape(-1, dim + 1)[:,1:]
    print("Getting new view...")
    f_new = new.view(numpy.float32)
    print("Creating split...")
    #Need better split method
    X_test = f_new[-10000:]
    X_train =f_new[:-10000]
    #X_train, X_test = train_test_split(new)
    #X_train = X_train[:10000,:]
    start = time.time()
    write_output(X_train, X_test, "Base0-1-2-3_merge", 'angular')
    end = time.time()
    print("DB done at " +str(end-start) + " seconds")
