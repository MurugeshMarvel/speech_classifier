import os
from os import listdir
from os.path import isfile, join
import theano
import numpy
import numpy as np
import csv
import pickle
import mfcc as mf

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
pik = open('dataset.pkl','wb')

def feature_extractor(wav_file):
	mfcc = mf.mfcc_generate(wav_file, length=16000)
	return mfcc

def prepare_lable(wav_path):
	index =0
	dir_index = []
	for i in range((len(wav_path)-1),-1,-1):
		if wav_path[i] == '/':
			dir_index.append(i)
			if i > index:
				index =i
	#print index
	return wav_path[dir_index[1]+1:dir_index[0]]
def parser(data_path):
	#labels = [f for f in listdir(data_path)]
	files = []
	for a,b,c in os.walk(data_path):
		for file in c:
			if file.endswith('.wav'):
				 files.append(os.path.join(a, file))
	return files



labvec = []
def lable2list(lable):
	if lable in labvec:
		pass
	else:
		labvec.append(lable) 

lables = []
feature_vector = []
for i in parser('audio_data/'):
	lables.append(prepare_lable(i))
print lables[1]
for i in range(0,len(lables)):
	lable2list(lables[i])
csvfile = open('dataset.csv','wb')
writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
inp = []
out = []
for i in parser('audio_data/'):
	lable_vector = int(labvec.index(prepare_lable(i)))
	out.append(lable_vector)
	feature = numpy.asarray(feature_extractor(i), dtype= numpy.float32)
	feature = feature.ravel()
	inp.append(feature)
	#writer.writerow([out,inp])
print numpy.shape(inp[2])
out= numpy.asarray(out)
inp = numpy.asarray(inp)
'''out = out.astype(numpy.float32)
inp = inp.astype(numpy.float32)'''
pickle.dump([inp, out], pik)
print 'finished extracting '

#def train_model()