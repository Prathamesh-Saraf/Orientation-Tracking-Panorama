import pickle
import sys
import time 
import pprint

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # need for python 3
  return d

dataset="1"

cfile = "C:/Users/prath/Desktop/ECE 276A/ECE276A_PR1/trainset/cam/cam" + dataset + ".p"
ifile = "C:/Users/prath/Desktop/ECE 276A/ECE276A_PR1/trainset/imu/imuRaw" + dataset + ".p"
vfile = "C:/Users/prath/Desktop/ECE 276A/ECE276A_PR1/trainset/vicon/viconRot" + dataset + ".p"
# cfile = "../trainset/cam/cam" + dataset + ".p"
# ifile = "../trainset/imu/imuRaw" + dataset + ".p"
# vfile = "../trainset/vicon/viconRot" + dataset + ".p"

ts = tic()
camd = read_data(cfile)
imud = read_data(ifile)
vicd = read_data(vfile)
toc(ts,"Data import")






