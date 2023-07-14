#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Written by Saliha and Murat Muradoglu
"""


import random
from itertools import permutations
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#%%
#Generate syllable structure based weighted distribution of types of onsets, nuclei and codas

def syll_struc():
    syll = "" 
    onset = ['C', '', 'CC']
    nucleus = ['V','D','V+']
    coda = ['C', '', 'CC']
    o = random.choices(onset, weights=(60, 30, 10), k=1)
    n = random.choices(nucleus, weights=(80, 10, 10), k=1)
    c = random.choices(coda, weights=(30, 60, 10), k=1)
    syll = o+n+c
    sylls = ''.join(syll)
    return sylls

#%%#Define Sonority metric for phoneme inventory
const_val={'p': 1, 't': 1, 'k': 1, 'b':2, 'd':2,'g':2,'ʧ':3, 'ʒ':4,'f':5, 's':5, 'ʃ':5,'v':6, 'n':7,'m':7,'ŋ':7,'l':8,'r':9,'w':10,'j':10}

# Calculate all possible permutations of 'CC'
A = ["p", "t", "k","b", "d", "g", "ʧ", "ʒ", "f", "s", "ʃ", "v", "m", "n", "ŋ", "l", "r", "w", "j"]
B = list(permutations(A,2))

# Convert tuple list to string list
def CC(B):
    return [''.join(i) for i in permutations(A,2)]
J=CC(B)



#%%# Replace phonemes with value according to Sonority Metric
liste=[]
for row in J:
    listy=[]
    for c in row:
        listy.append(const_val.get(c))
    liste.append(listy)
    
#%%#Calculate distance between two points/phonemes, allowing for direction
def distance(p):
    x1 = p[0]
    x2 = p[1]
    return x1-x2

#Create list of distance values for all permutations of 'CC'
sonor=[]
for item in liste:
    sonor.append(int(distance(item)))
    
#Convert list of distance values to dictionary (i.e. distance value has mapping to 'CC')
dictionary = dict(zip(J, sonor))

#Filter for permutations with distance greater than 3 for coda
d = dict((k, v) for k, v in dictionary.items() if v >= 3)

dictkeys_coda = list(d.keys())

#Filter for permutations with distance less than -3 for onset
e = dict((k, v) for k, v in dictionary.items() if v <= -3)

dictkeys_onset = list(e.keys())

#%%##Replace syllable structure with phoneme inventory

def syll_fill(syll):
    v = ["a","e","i","o","u"]
    d = ["ai","ei","ou"]
    c = ["p", "t", "k","b", "d", "g", "ʧ", "ʒ", "f", "s", "ʃ", "v", "m", "n", "ŋ", "l", "r", "w", "j"]
    
    if 'CC' in syll:
        cccount = syll.count('CC')
        ccinter = ["CC"] * cccount
        ccrand_c = random.choices(dictkeys_coda, k=cccount)
        ccrand_o = random.choices(dictkeys_onset, k=cccount)
        if syll.startswith('CC'):
            for i in range(cccount):
                syll=syll.replace(ccinter[i],ccrand_o[i],1)
        else:
            for i in range(cccount):
                syll=syll.replace(ccinter[i],ccrand_c[i],1)
        vcount = syll.count('V')
        ccount = syll.count('C')
        dcount = syll.count('D')
        vinter = random.choices('V', k=vcount)
        cinter = random.choices('C', k=ccount)
        dinter = random.choices('D', k=dcount)
        vrand = random.choices(v, k=vcount)
        crand = random.choices(c, k=ccount)
        drand = random.choices(d, k=dcount)
        for k in range(vcount):
            syll = syll.replace(vinter[k], vrand[k], 1)
        for j in range(ccount):
            syll = syll.replace(cinter[j], crand[j], 1)
        for l in range(dcount):
            syll = syll.replace(dinter[l], drand[l], 1)
    else:
        vcount = syll.count('V')
        ccount = syll.count('C')
        dcount = syll.count('D')
        vinter = random.choices('V', k=vcount)
        cinter = random.choices('C', k=ccount)
        dinter = random.choices('D', k=dcount)
        vrand = random.choices(v, k=vcount)
        crand = random.choices(c, k=ccount)
        drand = random.choices(d, k=dcount)
        for k in range(vcount):
            syll = syll.replace(vinter[k], vrand[k], 1)
        for j in range(ccount):
            syll = syll.replace(cinter[j], crand[j], 1)
        for l in range(dcount):
            syll = syll.replace(dinter[l], drand[l], 1)
    return syll


#%%##Create words by joining syllable structures and replacing structures with phonemes, exclude invalid vowels: these would be treated as long vowels. 
def word():
    words = []
    invalidV = ["aa", "oo", "uu", "ii", "ee"]
    for _ in range(25000):
        num_sylls = np.random.randint(1,10)
        s =''
        for j in range(num_sylls):
            s+=(syll_struc())
        sf = syll_fill(s)
        if any(V in sf for V in invalidV):
            continue  
        else: 
            words.append(sf)
    return words

w = word()

#Convert to set & back to ensure no duplicate words
WordSet = list(set(w))

#Definte Gaussian
def gaussian(x,mu=0,sig=1):
    return np.exp(-0.5*((x-mu)/(sig))**2);

#%%##Create list of word lengths
LengthVector = []

for word in WordSet:
    LengthVector.append(len(word))
#start from 1 as there are no 0 length words
x = np.linspace(1,np.max(LengthVector),10000);

# Estimate Kernel Density
kernel = stats.gaussian_kde(LengthVector)
LengthPDF = kernel(x)
LengthCDF = np.cumsum(LengthPDF)

plt.hist(LengthVector,bins=21,density=True)
plt.plot(x,LengthPDF,'r')
plt.legend(['Histogram', 'Kernel Estimate'])

#%%##%% Define Sampling Distribution. 
# Mean 8 letters, std dev 4
Sampling_PDF = gaussian(x,8,3)
Norm= np.trapz(Sampling_PDF); #Normalize our gaussian so integral == 1
Sampling_CDF = np.cumsum(Sampling_PDF/Norm)

plt.plot(x,Sampling_PDF)
plt.plot(x,LengthPDF/np.max(LengthPDF),'r') # Meaningless normalization. Just so we can get nice plot
plt.legend(['Sampling PDF','Length PDF'])

#%% Sample from Gaussian distribution
# How many samples to take
#NumberofTrials should be < WordSet so there are no duplicates

NumberOfTrials = 5000
unVals = np.random.uniform(0,1,NumberOfTrials)

# Round sample length to nearest integer
sampleLengths = []
sampleIndex = []

# Create copies of WordSet and LengthVector
# This is because we will remove sampled words from these as we go
LengthVectorCopy = LengthVector.copy()
WordSetCopy = WordSet.copy()

sampleset = []

for unVal in unVals:
    SampleLength = np.round(x[np.argmin(np.abs(Sampling_CDF-unVal))])
    sampleLengths.append(SampleLength)
    
    # Find index of length vector where we match sample length
    idx = np.where(LengthVectorCopy == SampleLength)
    
    # We check if there are any more words of the required SampleLength
    if np.any(LengthVectorCopy==SampleLength):
        
        idx = np.random.choice(idx[0], 1).tolist()[0]
        # Get sampled word
        sampleset.append(WordSetCopy[idx])
        
        # remove the sampled word from length + WordSet
        WordSetCopy.pop(idx)
        LengthVectorCopy.pop(idx)
    
    else:
        print("No words to sample of length %d"%SampleLength)
    

plt.hist(sampleLengths,bins=31)
plt.legend(['Sampled Word Lengths'])

# Word Lengths left can be seen
print(len(LengthVectorCopy))
# Words left 
print(len(WordSetCopy))

#%%#
#test sampling distribution

SL = []

for wugs in sampleset:
    SL.append(float(len(wugs)))

plt.hist(SL,bins=31)
plt.legend(['Sampled Word Lengths'])

print(len(sampleLengths))
print(len(SL))
print(SL == sampleLengths)

def Average(lst): 
    return sum(lst) / len(lst) 

print(Average(sampleLengths))
print(Average(SL))

print(SL[55])
print(sampleset[55])
print(sampleLengths[55])


            
#%%# 
output=open('sample_lexicon.txt','w')

for element in sampleset:
     output.write(element)
     output.write('\n')
output.close()