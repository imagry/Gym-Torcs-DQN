import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
from collections import deque
import os
import pickle
import sg_filter
import math

def sliding_win(arr, win_len):
     res = []
     for i in range(len(arr)):
	s = 0
	for j in range(win_len):
	    d = min(max(i + j - win_len/2, 0), len(arr)-1)
 	    s += arr[d]
	s /= win_len
	res.append(s)
     return np.array(res)
	

def rescale(arr):
     rmin = arr.min()
     rmax = arr.max()
     return (arr - rmin)/(rmax-rmin)

#a_0_list = pickle.load(open("a_0_list.bin", "rb"))

curr_dir = ""#""/home/sergey/repo/gym-torcs/last_stat/"
tstr = "_18_15_58_30238"
	#"_17_02_43_100161"
	#"_interrupt_126381"
q_loss_list = pickle.load(open(curr_dir + "q_loss_list" + tstr +".bin", "rb"))
track_list = pickle.load(open(curr_dir + "track_list" + tstr +".bin", "rb"))
#yspeed_list = pickle.load(open(curr_dir + "yspeed_list" + tstr +".bin", "rb"))
reward_list = pickle.load(open(curr_dir + "reward_list" + tstr +".bin", "rb"))

stop = -1

for i in range(len(q_loss_list)) :
	if math.isnan(q_loss_list[i]) :
		stop = i
		break

reward_list = reward_list[0:stop:2]
q_loss_list  =q_loss_list [0:stop:2]
track_list  =track_list [0:stop:2]
#yspeed_list  =yspeed_list [0:stop:10]

win_len = 101
start = 1
end = len(q_loss_list)-win_len/2-1
reward_list = np.array(reward_list[start:end])
q_loss_list = np.array(q_loss_list [start:end])
track_list = np.array(track_list [start:end])
#yspeed_list = np.array(yspeed_list [start:end])

reward_list = sliding_win(reward_list, win_len)
q_loss_list = sliding_win(q_loss_list, win_len)
track_list = sliding_win(track_list, win_len)
#yspeed_list = sliding_win(yspeed_list, win_len)

q_loss_list = rescale(q_loss_list)
reward_list = rescale(reward_list)

plt.plot(np.array(range(len(reward_list))), reward_list, color="grey", linewidth=0.5, linestyle="-", label="reward")
plt.plot(range(len(q_loss_list)), q_loss_list, color="green", linewidth=0.5, linestyle="-", label="q_loss")
plt.plot(range(len(track_list)), track_list, color="red", linewidth=0.5, linestyle="-", label="track_list")
#plt.plot(range(len(yspeed_list)), yspeed_list, color="blue", linewidth=0.5, linestyle="-", label="yspeed_list")


plt.legend(loc='upper left')

plt.show()
