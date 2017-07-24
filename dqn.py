import numpy as np
import math
import cv2
from PIL import Image
import sys
import random
import os
import cPickle
from rambo_preprocessing import image_sequence_preprocessing
from gym_torcs import TorcsEnv
import caffe
from random import randint
import time
from collections import (deque, defaultdict)
from scipy.signal import argrelextrema
from torcs_config import change_track, track_list as t_list


#data duplicated in prototxt files:
BATCH_SIZE = 24#32--
BUFFER_SIZE = 70000#110000
STATE_HEIGHT = 192
STATE_WIDTH = 256
CHANNELS = 3
DISCR_A = 11
DELTA_A = .85
#data positions in tuple sample
i_s = 0
i_ns = 1
i_action = 2
i_reward = 3
i_done = 4
i_pos = 5
i_tid = 6
play = False #no training
current_dir = '/home/sergey/repo/gym-torcs/'

def sign(x):
    return float(x > 0) - float(x < 0)

def ParamCopy(dst, src):
    keys_s = set(src.keys())
    keys_d = set(dst.keys())
    intersection = keys_s & keys_d
    for k in intersection:
        if len(dst[k]) != len(src[k]):
            print 'param err', len(dst[k]), len(src[k])
            os.system('pkill torcs')
            quit()
        for i in range(len(dst[k])):
            if dst[k][i].data.shape != src[k][i].data.shape:
                print 'shape err', k, i, dst[k][i].data.shape, src[k][i].data.shape
                os.system('pkill torcs')
                quit()
            dst[k][i].data[...] = src[k][i].data


def SoftUpdate(dst, upd, tau):
    keys_u = set(upd.keys())
    keys_d = set(dst.keys())
    intersection = keys_u & keys_d
    for k in intersection:
        if len(dst[k]) != len(upd[k]):
            print 'param err', len(dst[k]), len(upd[k])
            os.system('pkill torcs')
            quit()
        for i in range(len(dst[k])):
            if dst[k][i].data.shape != upd[k][i].data.shape:
                print 'shape err', k, i, dst[k][i].data.shape, upd[k][i].data.shape
                os.system('pkill torcs')
                quit()
            temp = tau * upd[k][i].data
            # tau*upd[k][i].data + (1-tau)*dst[k][i].data
            dst[k][i].data[...] *= (1 - tau)
            dst[k][i].data[...] += temp

class ReplayBufferN(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch4Pos(self, batch_size, start_pos, track_id):
        batch = []
        valid = True
        assert start_pos >= 0
        assert start_pos + batch_size < self.num_experiences
        for ii in range(batch_size):
            if self.buffer[start_pos][i_pos] + ii != self.buffer[start_pos + ii][i_pos] \
                    or track_id != self.buffer[start_pos + ii][i_tid]:
                valid = False
                break
        if valid:
            for ii in range(start_pos, start_pos + batch_size):
                batch.append(self.buffer[ii])
        else:
            print "discontinue"
            return batch, False, -1, -1,

        if start_pos - batch_size >= 0 and self.buffer[start_pos - batch_size][i_tid] == track_id \
                    and self.buffer[start_pos - batch_size][i_pos] == self.buffer[start_pos][i_pos] - batch_size:
            print "continue n-steps  +++++++++++++++++++++++++++++",\
                start_pos + batch_size,"<--",start_pos,"next <--",start_pos - batch_size
            return batch, True, start_pos - batch_size, track_id
        else:
            print "stop n-steps  +++++++++++++++++++++++++++++", \
                start_pos + batch_size, "<--", start_pos
            return batch, True, -1, -1,

    def try_start(self, rstart, max_n_batches, batch_size):
        # shift start of batch so that max n-step size fit
        max_range = min(self.num_experiences - rstart, max_n_batches * batch_size)
        shift_range = max_range
        for ii in range(max_range):
            if self.buffer[rstart][i_pos] + ii != self.buffer[rstart + ii][i_pos] \
                    or self.buffer[rstart][i_tid] != self.buffer[rstart + ii][i_tid]\
                    or self.buffer[rstart + ii][i_reward] == 0:
                shift_range = ii
                break

        rstart = rstart + shift_range - batch_size

        # recheck batch and get reward variance
        var = 0
        success = True
        for ii in range(batch_size):
            if self.buffer[rstart][i_pos] + ii != self.buffer[rstart + ii][i_pos] \
                    or self.buffer[rstart][i_tid] != self.buffer[rstart + ii][i_tid] \
                    or self.buffer[rstart + ii][i_reward] == 0:
                success = False
                var = -1
                break
            elif ii < batch_size-1:
                var += np.abs(self.buffer[rstart + ii + 1][i_reward] - self.buffer[rstart + ii][i_reward])\
                       /(self.buffer[rstart + ii][i_reward] + 1e-2)
        var /= batch_size-1
        return success, rstart, var

    def getBatch(self, batch_size, max_n_batches, k_priority_try, n_steps = False,  priority = False, start_ind = 300, min_val = 1., max_val = 10.):
        # Randomly sample batch_size examples
        if n_steps:
            rstart = random.randint(0, self.num_experiences - batch_size)
            #priority sampling
            n_steps, rstart, variance = self.try_start(rstart, max_n_batches, batch_size)
            for ktry in range(0, k_priority_try-1):
                rstart_try = random.randint(0, self.num_experiences - batch_size)
                n_steps_try, rstart_try, variance_try = self.try_start(rstart_try, max_n_batches, batch_size)
                if n_steps_try and variance_try > variance:
                    n_steps = n_steps_try
                    rstart = rstart_try
                    variance = variance_try
                    print "priority switch"
            if n_steps:
                batch = []
                for ii in range(rstart, rstart + batch_size):
                    batch.append(self.buffer[ii])
                if rstart - batch_size >= 0 and self.buffer[rstart - batch_size][i_tid] == self.buffer[rstart][i_tid]\
                        and self.buffer[rstart - batch_size][i_pos] == self.buffer[rstart][i_pos] - batch_size:
                    print "finish n-steps batch***********************", rstart + batch_size, "<--",rstart,"<--next",rstart - batch_size
                    return batch, True, rstart - batch_size, self.buffer[rstart][i_tid]
                else:
                    print "finish n-steps batch***** simple",rstart - batch_size,"tstart", self.buffer[rstart][i_tid],\
                        "tnew", self.buffer[max(rstart - batch_size, 0)][i_pos]
                    return batch, True, -1, -1

        if not n_steps and (not priority or self.num_experiences < batch_size):
            print "simple batch"
            if self.num_experiences < batch_size:
                return random.sample(self.buffer, self.num_experiences), False, -1, -1
            else:
                return random.sample(self.buffer, batch_size), False, -1, -1
        else:
            print "priority batch"
            P = np.empty(self.num_experiences)
            P.fill(min_val)
            for i in range(self.num_experiences):
                w_p = self.buffer[i][i_pos]
                alpha = min(max((w_p - start_ind)/1000., 0.), 1.)
                P[i] += (max_val - min_val)*alpha
            s = np.sum(P)
            P /= s
            ch_arr = np.random.choice(range(self.num_experiences), batch_size, False, P)
            for ii in ch_arr:
                batch.append(self.buffer[ii])
            return batch, False, -1, -1

    def get(self, i):
        return self.buffer[i]

    def size(self):
        return self.buffer_size

    def add(self, state_i, action, reward, new_state_i, done, w_p, tid, p1, p2):
        state = []
        new_state = []
        for i in range(CHANNELS) :
            flag, istate = cv2.imencode("1.jpg", state_i[i].reshape(STATE_HEIGHT, STATE_WIDTH), [cv2.IMWRITE_JPEG_QUALITY, 80])
            state.append(istate)
            flag, inew_state = cv2.imencode("2.jpg", new_state_i[i].reshape(STATE_HEIGHT, STATE_WIDTH), [cv2.IMWRITE_JPEG_QUALITY, 80])
            new_state.append(inew_state)
            experience = (state, new_state, action, reward, done, w_p, tid, p1, p2)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def push(self, state, action, reward, new_state, done, w_p, tid, p1, p2):
        experience = (state, new_state, action, reward, done, w_p, tid, p1, p2)
        self.buffer.append(experience)
        self.num_experiences += 1

    def count(self):
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

    def size_reduce(self, max_size):
        while self.num_experiences >= max_size:
            self.buffer.popleft()
            self.num_experiences -= 1

def save_state(a_0_list, a_1_list, q_loss_list, reward_list, track_list, yspeed_list, tstr):
    cPickle.dump(a_0_list, open( "a_0_list_" + tstr +".bin", "wb" ) )
    cPickle.dump(q_loss_list, open( "q_loss_list" + tstr +".bin", "wb" ) )
    cPickle.dump(reward_list, open( "reward_list" + tstr +".bin", "wb" ) )
    cPickle.dump(track_list, open("track_list" + tstr +".bin", "wb"))
    cPickle.dump(yspeed_list, open("yspeed_list" + tstr +".bin", "wb"))

def save_nets(q_act_net, q_target_net, ind, buffer, str_add = ""):
    print 'saving nets'
    q_act_net.save('q_solver_' + str(ind) +'.caffemodel')
    q_target_net.save('qq_target_' + str(ind) +'.caffemodel')
    print 'saving replay_buffer buffer'
    save_replay(buffer, str_add)
    print 'replay_buffer buffer saved'

def save_replay(buffer, str_add):
    cPickle.dump(buffer, open("buffer"+str_add+".replay", "wb"))

def load_replay():
    return  cPickle.load(open("buffer.replay", "rb"))

a_0_list = []
a_1_list = []
q_loss_list = []
reward_list = []
track_list = []
yspeed_list = []
step = 0
replay_buffer = ReplayBufferN(BUFFER_SIZE)
q_act_net = None
q_target_net = None

#make input image
def make_state(images, sequence_length):
    if len(images) < sequence_length + 1:
        return None
    image_differences = image_sequence_preprocessing(images, sequence_length = sequence_length)
    return np.array(image_differences)

def unpack(jpg_slice):
    return np.array(cv2.imdecode(jpg_slice,  cv2.CV_LOAD_IMAGE_GRAYSCALE))

#action value to action index
def a2ind(action, discr_a, delta_a):
    dh = (discr_a - 1)/2
    return int(np.rint(dh * action / delta_a + dh))

#action index to action value
def ind2a(qind, discr_a, delta_a):
    hd = float((discr_a - 1)/2)
    return (float(qind) - hd)*delta_a/hd

#driving agent using dqn
def qchoice(net, state_blob, sequence_length, discr_a, delta_a):
    if state_blob is None:
        return None
    net.blobs['state'].data[...] = state_blob.reshape(1, sequence_length, STATE_HEIGHT, STATE_WIDTH)
    q_a = net.forward()['q_action']
    if not play:
        print 'curr q_action', q_a.reshape(discr_a)
    max_Q = np.argmax(q_a.reshape(discr_a))
    if np.sum(np.abs(q_a)) < 1e-3:
        max_Q = discr_a/2
    action = ind2a(max_Q, discr_a, delta_a)
    return action

#action smoother
def l2_var_spatial_reg(net, lambda_reg):
    a = np.array(net.blobs['q_action'].data[...])
    I = np.array(net.blobs['fc2q'].data[...])
    w = np.array(net.params['fc_q'][0].data[...])
    b = np.array(net.params['fc_q'][1].data[...])
    batch_size = a.shape[0]
    v_num = w.shape[0]
    v_size = w.shape[1]
    grad_b = np.zeros(v_num)
    grad_w = np.zeros((v_num, v_size))
    # grad w = sum_I (wI+b - a)I^t
    # grad b = sum_I (wI+b - a)

    for n in range(batch_size):
        I_n = I[n, :]
        for i in range(v_num):
            a_c = a[n, i]

            if i > 0:
                delta = a_c - a[n, i-1]
                grad_b[i] += delta
                grad_w[i, :] += delta*I_n

            if i > 1:
                delta = a_c - a[n, i-2]
                grad_b[i] += .5*delta
                grad_w[i, :] += .5*delta*I_n

            if i > 2:
                delta = a_c - a[n, i-3]
                grad_b[i] += .25*delta
                grad_w[i, :] += .25*delta*I_n

            if i > 3:
                delta = a_c - a[n, i-4]
                grad_b[i] += .125*delta
                grad_w[i, :] += .125*delta*I_n

            if i < v_num-1:
                delta = a_c - a[n, i + 1]
                grad_b[i] += delta
                grad_w[i, :] += delta * I_n

            if i < v_num-2:
                delta = a_c - a[n, i + 2]
                grad_b[i] += .5*delta
                grad_w[i, :] += .5*delta * I_n

            if i < v_num-3:
                delta = a_c - a[n, i + 3]
                grad_b[i] += .25*delta
                grad_w[i, :] += .25*delta * I_n

            if i < v_num-4:
                delta = a_c - a[n, i + 4]
                grad_b[i] += .125*delta
                grad_w[i, :] += .125*delta * I_n

    nb = np.max(np.abs(b))
    nw = np.max(np.abs(w))

    ngb = np.max(np.abs(grad_b))+1e-8
    ngw = np.max(np.abs(grad_w))+1e-8

    lambda_eff_b = nb / ngb * lambda_reg
    lambda_eff_w = nw / ngw * lambda_reg

    w -= lambda_eff_w * grad_w
    b -= lambda_eff_b * grad_b

    net.params['fc_q'][0].data[...] = w
    net.params['fc_q'][1].data[...] = b

#second derivative regularzier
def d2_var_spatial_reg(net, lambda_reg):
    a = np.array(net.blobs['q_action'].data[...])
    I = np.array(net.blobs['fc2q'].data[...])
    w = np.array(net.params['fc_q'][0].data[...])
    b = np.array(net.params['fc_q'][1].data[...])
    batch_size = a.shape[0]
    v_num = w.shape[0]
    v_size = w.shape[1]
    assert I.shape[1] == v_size
    grad_b = np.zeros(v_num)
    grad_w = np.zeros((v_num, v_size))
    # grad w = sum_I (wI+b - a)I^t
    # grad b = sum_I (wI+b - a)

    for n in range(batch_size):
        I_n = I[n, :]
        for i in range(v_num):
            a_c = a[n, i]
            if i > 1 and i < v_num-1:
                #d2 >= 0
                d2 = a_c - .5*(a[n, i-1] + a[n, i+1])
                delta = -d2*(d2 < 0)
                grad_b[i] += delta
                grad_w[i, :] += delta*I_n
            if i < v_num - 2:
                d2 = a[n, i + 1] - .5*(a_c + a[n, i + 2])
                delta = d2 * (d2 < 0)
                grad_b[i] += delta
                grad_w[i, :] += delta*I_n
            if i > 2:
                d2 = a[n, i - 1] - .5*(a[n, i - 2] + a_c)
                delta = d2 * (d2 < 0)
                grad_b[i] += delta
                grad_w[i, :] += delta*I_n
    nb = np.max(np.abs(b))
    nw = np.max(np.abs(w))

    ngb = np.max(np.abs(grad_b))+1e-8
    ngw = np.max(np.abs(grad_w))+1e-8

    lambda_eff_b = nb / ngb * lambda_reg
    lambda_eff_w = nw / ngw * lambda_reg

    w -= lambda_eff_w * grad_w
    b -= lambda_eff_b * grad_b

    net.params['fc_q'][0].data[...] = w
    net.params['fc_q'][1].data[...] = b

#make one training iteration of solver
def solver_step_pos(q_target_net, new_states, solver, states, gamma, rewards, actions, dones,
                    discr_a, delta_a, batch_size, n_steps, n_step_continued, Qlast, lambda_spatial_q):

    #mirroring augmentaion
    if(random.random() < .5):
        states = states[:,:,::-1]
        new_states = new_states[:, :, ::-1]
        actions = -1*actions

    #target network for y regression value
    q_target_net.blobs['state'].data[...] = new_states
    output4q = q_target_net.forward()['q_action']

    #current net for initial regression value
    solver.net.blobs['state'].data[...] = states
    solver.net.forward()
    target = np.copy(solver.net.blobs['q_action'].data[...])

    if n_steps :
        # n-step regression preparation
        alpha = 1.
        print "n-steps solver***********************", "cont", n_step_continued
        for k in reversed(range(batch_size)):
            assert output4q[k, :].size == discr_a
            assert target[k, :].size == discr_a
            action_ind = a2ind(actions[k], discr_a, delta_a)
            if k == batch_size-1 and not n_step_continued:
                Qlast = np.max(output4q[k, :].reshape(discr_a))
            else:
                Qminus = np.max(output4q[k, :].reshape(discr_a))
                Qlast = alpha*max(Qminus - Qlast, 0) + Qlast
            assert action_ind in range(discr_a), " action_ind %r actions[k] %r k %r" % (action_ind, actions[k], k)
            if dones[k] or rewards[k] == 0:
                target[k, action_ind] = rewards[k]
            else:
                target[k, action_ind] = rewards[k] + gamma * Qlast

            Qlast = rewards[k] + gamma * Qlast
    else:
        # 1-step regression preparation
        print "batch solver ---------------------------- "
        for k in range(batch_size):
            assert output4q[k, :].size == discr_a
            assert target[k, :].size == discr_a
            Qa = np.max(output4q[k, :].reshape(discr_a))
            action_ind = a2ind(actions[k], discr_a, delta_a)
            assert action_ind in range(discr_a), " action_ind %r actions[k] %r k %r" % (action_ind, actions[k], k)
            if dones[k] or rewards[k] == 0:
                target[k, action_ind] = rewards[k]
            else:
                target[k, action_ind] = rewards[k] + gamma * Qa
    #fill solver net
    solver.net.blobs['label'].data[...] = target.reshape(batch_size, discr_a, 1, 1)
    solver.net.blobs['state'].data[...] = states
    #solver iteration
    solver.step(1)
    if lambda_spatial_q > 0:
        l2_var_spatial_reg(solver.net, lambda_spatial_q)
        #d2_var_spatial_reg(solver.net, lambda_spatial_q)
    return Qlast

#train net on batch from replay buffer
def train_on_batch(batch, q_target_net, solver, discr_a, delta_a,
                   batch_size, gamma, n_steps, n_step_continued, Qlast, lambda_spatial_q):
    jpeg_states = np.asarray([e[i_s] for e in batch])
    jpeg_new_states = np.asarray([e[i_ns] for e in batch])
    actions = np.asarray([e[i_action] for e in batch])
    rewards = np.asarray([e[i_reward] for e in batch])
    dones = np.asarray([e[i_done] for e in batch])

    assert batch_size == BATCH_SIZE
    states = np.zeros((BATCH_SIZE, CHANNELS, STATE_HEIGHT, STATE_WIDTH))
    new_states = np.zeros((BATCH_SIZE, CHANNELS, STATE_HEIGHT, STATE_WIDTH))

    for i in range(batch_size):
        for j in range(CHANNELS):
            states[i, j, :] = unpack(jpeg_states[i][j])
            new_states[i, j, :] = unpack(jpeg_new_states[i][j])

    Qlast = solver_step_pos(q_target_net, new_states, solver, states, gamma,
                            rewards, actions, dones, discr_a, delta_a, batch_size,
                            n_steps, n_step_continued, Qlast, lambda_spatial_q)

    q_loss = np.sum(
        solver.net.blobs['l2_loss'].data) / (batch_size*discr_a)
    return q_loss, Qlast

def main():
    global play, replay_buffer, q_act_net, q_target_net, step
    #sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    tstr = time.strftime('_%H_%M_%S_')

    first_run = False #don't use saved nets/buffer
    reset_buffer = True #don't use saved buffer


    #dqn params
    GAMMA =0.99
    TAU = .0001#.0001

    #exploration noise params
    act_noise_init = 0.5 #.75 #.25
    act_noise_final = .01  # .25
    act_noise_interval = 100000
    rnd_range = 1

    #lag augmentation
    packet_lost = 0 #0.01

    #action smoothing augmentation
    lambda_spatial_q = 0
    action_smoother = .33
    action_limiter = .33

    episode_count = 10000
    max_steps = 5000
    save_in_iters = 15000
    #100000

    #start training after accumulating train_start_num samples
    train_start_num = 1 * BATCH_SIZE

    caffe.set_mode_gpu()
    caffe.set_device(0)

    #balance track samples in replay buffer
    track_balance = .9

    #n-steps dqn, steps=max_n_batches*batch size
    max_n_batches = 16

    # use n-steps dqn for this ratio, rest 1-step
    n_step_ratio = .75

    #priority buffer
    k_priority_try = 2

    if n_step_ratio > 0 and max_n_batches > 0:
        n_steps_dqn = True
    else:
        n_steps_dqn = False

    #average speed
    c_speed = 35  # 50
    #speed variance
    delta_speed = 5  # 7.5
    #switch speed in
    swith_count_min = 50


    #target frame time
    frame_rate = .4
    #fail if lag is more then
    t_delta_fail = frame_rate*1.75

    #for rebound handling
    rebound_count_max = 5

    start_run = 50
    # for error handling
    after_start_check = 100
    max_errors = 50

    #solver for current net
    if not play:
        critic_solver = caffe.get_solver(current_dir + 'resnet_torcs/dqn_critic_solver.prototxt')
    if first_run:
        # target net:
        q_target_net = caffe.Net(current_dir + 'resnet_torcs/critic_batch_dqn.prototxt',
                                 current_dir + 'r18nb.caffemodel', caffe.TEST)
        # current net:
        q_act_net = caffe.Net(current_dir + 'resnet_torcs/critic_deploy_dqn.prototxt', caffe.TEST)
        if not play:
            ParamCopy(critic_solver.net.params, q_target_net.params)
        ParamCopy(q_act_net.params, q_target_net.params)
    else:
        #target net:
        q_target_net = caffe.Net(current_dir + 'resnet_torcs/critic_batch_dqn.prototxt', 'qq_target.caffemodel', caffe.TEST)
        #current net:
        q_act_net = caffe.Net(current_dir + 'resnet_torcs/critic_deploy_dqn.prototxt', 'q_solver.caffemodel',caffe.TEST)
        if not play:
            ParamCopy(critic_solver.net.params, q_act_net.params)
        if not play and not reset_buffer:
            print 'loading replay_buffer buffer'
            replay_buffer = load_replay()
            replay_buffer.size_reduce(BUFFER_SIZE)
            print 'replay_buffer buffer loaded'

    print 'models loaded ***************************'

    if not play:
        assert q_target_net.blobs['state'].data.shape[0] == BATCH_SIZE
        assert q_act_net.blobs['state'].data.shape[0] == 1
        assert critic_solver.net.blobs['state'].data.shape[0] == BATCH_SIZE

        assert q_target_net.blobs['state'].data.shape[1] == CHANNELS
        assert q_act_net.blobs['state'].data.shape[1] == CHANNELS
        assert critic_solver.net.blobs['state'].data.shape[1] == CHANNELS

        assert q_target_net.blobs['q_action'].data.shape[1] == DISCR_A
        assert q_act_net.blobs['q_action'].data.shape[1] == DISCR_A
        assert critic_solver.net.blobs['q_action'].data.shape[1] == DISCR_A

    max_reached_step = 150#used for track balance
    images_history = [] #used for input image
    step = 0 #total number of simulation steps
    save_count = 0 #used for saving nets/buffer
    n_batch = 0 #used for n-steps

    q_loss = 0 #main loss
    # Generate a Torcs environment
    env = TorcsEnv(vision=True, throttle=False, observer=False)
    time_start = time.time()
    track_id = 0 #track

    #n-step temp vars
    n_steps_cont_from_prev = False
    prev_start_pos = -1
    prev_track_id = -1
    Qlast = -1
    episod_steps = 0
    n_steps_used = 0
    batches_used = 0

    #for error failure
    rest_fail = 0

    rebound_events = 0

    for i in range(episode_count):
        #balance tracks
        if episod_steps >= max_reached_step*track_balance:
            track  = t_list[track_id]
            change_track("/usr/local/share/games/torcs/config/raceman/quickrace.xml", track)
            print "Track: ", track, "track_id", track_id
            episod_steps = 0

        print("Episode : " + str(i))
        ob = env.reset(relaunch=True)

        s_t = None #input image
        total_reward = 0.

        #for randomizing velocity
        switch_count = swith_count_min + random.randint(0, swith_count_min)

        #for handling out-of-lane
        rebound = False
        rebound_count = 0
        track_pos = 0
        error_count = 0
        act_prev = np.array([0.])
        t_delta = 0

        for j in range(max_steps):
            max_reached_step = max(max_reached_step, j)
            a_t = np.array([0.])#action
            skip_state = False
            error_present = False

            #exploration noise params
            act_noise = act_noise_init + (act_noise_final-act_noise_init)*min(step*1./act_noise_interval, 1.)
            rnd_noise = 1
            if rnd_range > 1:
                rnd_noise = int((rnd_range + 1)*max(1., float(act_noise_interval - step)/act_noise_interval))

            #get action =======================================================
            if s_t is None:
                action_index = random.randrange(DISCR_A)
                print '----------Random Action---------- action_index', action_index
                a_t[0] = ind2a(action_index, DISCR_A, DELTA_A)
            else:
                a_t[0] = qchoice(q_act_net, s_t, CHANNELS, DISCR_A, DELTA_A)
                #apply exploration noise
                if not play and random.random() <= act_noise:
                    ind = a2ind(a_t[0], DISCR_A, DELTA_A)
                    r = 1
                    if rnd_noise > 1:
                        r = randint(1, rnd_noise)
                    ind += randint(-r, r)
                    ind = min(max(ind, 0), DISCR_A-1)
                    a_t[0] = ind2a(ind, DISCR_A, DELTA_A)

            #if still no action use random
            if  a_t is None:
                action_index = random.randrange(DISCR_A)
                print 'rnd action_index', action_index
                a_t[0] = ind2a(action_index, DISCR_A, DELTA_A)

            #starting area
            if j < start_run:
                a_t[0] = 0

            #action limiter
            if not play and abs(a_t[0]) > DELTA_A/2 and random.random() < action_limiter:
                ind = a2ind(a_t[0], DISCR_A, DELTA_A)
                dind = ind - DISCR_A/2
                if dind > (DISCR_A-1)/4:
                    dind = (DISCR_A-1)/4
                if dind < -(DISCR_A-1)/4:
                    dind = -(DISCR_A-1)/4
                a_t[0] = ind2a(dind + DISCR_A/2, DISCR_A, DELTA_A)

            #save action
            a_0_list.append(a_t)

            #fail on render delay
            if  not play and t_delta > t_delta_fail and i > rest_fail + 10 and j >= after_start_check:
                error_present = True
                if error_count >= max_errors/2:
                    print 'delta fail **************************'
                    rest_fail = i
                    time_start = time_end
                    break
                else:
                    error_count += 1

            #randomize speed
            if (j % switch_count and not play) == 0:
                tag_speed_rnd = c_speed - delta_speed + random.uniform(0, delta_speed*2)
            else:
                tag_speed_rnd = c_speed


            #render delay compensation
            if t_delta > frame_rate:
                tag_speed = frame_rate/t_delta*tag_speed_rnd
            else:
                tag_speed = tag_speed_rnd

            #handle out-of-lane event
            if rebound:
                rebound_count = rebound_count_max
            else:
                rebound_count = max(0, rebound_count - 1)
            if (rebound_count > rebound_count_max/2 and abs(track_pos) > .7) or rebound:
                angle = - observation.angle
                if angle * track_pos > 0 and abs(angle) > .2 :
                    a_t[0] = -sign(track_pos)*4*DELTA_A/5
                if angle * track_pos > 0 and abs(angle) <= .2 :
                    a_t[0] = -sign(track_pos)*2*DELTA_A/5
                if angle*track_pos < 0 and abs(angle) <= .15:
                    a_t[0] = -sign(track_pos)*DELTA_A/5
                if angle*track_pos < 0 and abs(angle) > .15:
                    a_t[0] = 0
                if angle*track_pos < 0 and abs(angle) >= .35:
                     a_t[0] = sign(track_pos)*DELTA_A/5
                tag_speed = min(tag_speed, 20)
                print "############ rebound, action",  a_t[0], "V angle", angle, "###############"


            #smooth action
            if  not play and action_smoother > 0 and random.random() < action_smoother:
                ind_prev = a2ind( act_prev[0], DISCR_A, DELTA_A)
                ind = a2ind(a_t[0], DISCR_A, DELTA_A)
                if abs(ind - ind_prev) > 1:
                    print "smooth ind", ind, "->",np.rint(.5*(ind_prev + ind))
                    ind = int(.5*(ind_prev + ind))
                    a_t[0] = ind2a(ind, DISCR_A, DELTA_A)

            a_act = a_t
            #lag augemntaion
            if not play and random.random < packet_lost and t_delta < frame_rate:
                 a_act = act_prev

            #===================== main enviroment step =========================================
            obs0 = time.time()
            prev_rebound = rebound
            observation, r_t, done, rebound, _ = env.step(a_act, tag_speed)
            curr_time = time.time()
            t_delta = curr_time - time_start
            time_start = curr_time
            #====================================================================================
            if rebound and not prev_rebound:
                rebound_events += 1
            print 't_delta', t_delta, "step", j, "step time", curr_time-obs0, "tag_speed_rnd",tag_speed_rnd, "rebound_events", rebound_events
            if rebound:
                r_t = 0
            if prev_rebound and r_t == 0:
                skip_state = True

            #speed failure, could be moved to gym_torcs
            if observation.speedX < .01 and j >= after_start_check and t_delta < t_delta_fail:
                skip_state = True
                error_present = True
                r_t = 0
                if error_count >= max_errors :
                    print 'speed too slow fail, speed', 300*observation.speedX, '**************************'
                    break
                else:
                    error_count += 1

            #make state ========================================================
            image = observation.img
            images_history.append(image)
            while len(images_history) > CHANNELS + 1:
                images_history.pop(0)
            s_t1 = make_state(images_history, CHANNELS)
            track_pos = observation.trackPos

            #save stat
            reward_list.append(r_t)
            track_list.append(track_pos)
            yspeed_list.append(observation.speedY)

            #store data into replay buffer ======================================
            do_store = not play and s_t is not None and s_t1 is not None and not skip_state
            if do_store :
                print 'add data, action', a_t[0],  'reward ', r_t
                w_p = j
                replay_buffer.add(s_t, a_t, r_t, s_t1, done, w_p, track_id, -1, -1)
                print '***** stored: track_pos', track_pos, 'angle', observation.angle,\
                    'max_step', max_reached_step, 'Episode', i
            elif not play:
                print 'skipped state track_pos', track_pos, 'angle', observation.angle,\
                    'max_step', max_reached_step, 'Episode', i

            #training  ======================================
            if  not play and replay_buffer.num_experiences > train_start_num:
                #get batch using n-steps if previous batch was using n-step
                use_n_steps_now = n_steps_dqn
                if n_batch >= max_n_batches:
                    use_n_steps_now = False
                    n_batch = 0
                if n_steps_cont_from_prev and use_n_steps_now and max_n_batches > 1:
                    assert prev_start_pos >= 0
                    batch, n_steps_collected, prev_start_pos, prev_track_id =\
                        replay_buffer.getBatch4Pos(BATCH_SIZE, prev_start_pos, prev_track_id)
                    n_step_continued = n_steps_collected
                else:
                    n_step_continued = False
                if n_steps_used >= n_step_ratio*batches_used and not n_step_continued:
                    use_n_steps_now = False

                #get batch if previous batch was *not* using n-step
                if not n_step_continued:
                    batch, n_steps_collected, prev_start_pos, prev_track_id =\
                        replay_buffer.getBatch(BATCH_SIZE, max_n_batches, k_priority_try, n_steps=use_n_steps_now)

                #net training =============
                q_loss, Qlast = train_on_batch(batch, q_target_net, critic_solver, DISCR_A, DELTA_A,
                          BATCH_SIZE, GAMMA, n_steps_collected, n_step_continued, Qlast, lambda_spatial_q)

                #update n-step vars
                if n_steps_collected:
                    n_batch += 1
                    n_steps_used += 1
                else :
                    n_batch = 0
                batches_used += 1
                n_steps_cont_from_prev = n_steps_collected and prev_start_pos >= 0

                # target update ==============
                SoftUpdate(q_target_net.params,
                           critic_solver.net.params, TAU)
                ParamCopy(q_act_net.params,
                          critic_solver.net.params)
                save_count += 1

            #save loss
            if not play:
                q_loss_list.append(q_loss)

            #update local vars
            s_t = s_t1
            act_prev = a_t
            if done:
                s_t = None
            if not error_present:
                error_count = max(0, error_count - 1)
            total_reward += r_t
            episod_steps += 1
            step += 1
            if done:
                break

        #save nets and buffer
        if not play and save_count >= save_in_iters:
            print "start save", save_count, step
            save_count = 0
            save_nets(q_act_net, q_target_net, step, replay_buffer)
            save_state(a_0_list, a_1_list, q_loss_list, reward_list, track_list, yspeed_list, tstr + str(step))
        track_id = (track_id + 1)%len(t_list)
        print("TOTAL REWARD @ " + str(i) +
              " -th Episode  :  " + str(total_reward))
        print("Total Step: " + str(step))
        print("")
    print("Finishing torcs.")
    env.end()  # This is for shutting down TORCS
    #save nets and buffer
    if not play:
        save_state(a_0_list, a_1_list, q_loss_list, reward_list, track_list, yspeed_list, tstr + str(step))
        save_nets(q_act_net, q_target_net, step, replay_buffer, "_finished")
    print 'Finish'

try:
    if __name__ == "__main__":
        main()
except KeyboardInterrupt:
    os.system('pkill torcs')
    if not play:
        save_state(a_0_list, a_1_list, q_loss_list, reward_list, track_list, yspeed_list, "_interrupt_" + str(step))
        save_nets(q_act_net, q_target_net, step, replay_buffer, "_interrupt" + str(step))
    raise
except:
    if not play:
        save_state(a_0_list, a_1_list, q_loss_list, reward_list, track_list, yspeed_list, "_interrupt" + str(step))
        save_nets(q_act_net, q_target_net, step, replay_buffer, "_interrupt" + str(step))
    raise

