import collections as col
import copy
import numpy as np
import math
import os
import time

import cv2
from gym import spaces

import snakeoil3_gym as snakeoil3

SCREEN_WIDTH = 256
SCREEN_HEIGHT = 192

def indicator(x):
    if x > 0:
        return 1.
    else:
        return 0.

class TorcsEnv:
    terminal_judge_start = 500  # Speed limit is applied after this step
    termination_limit_progress = 5  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 45
    initial_reset = True


    def __init__(self, vision=False, throttle=False, gear_change=False, observer = True, termination = True):
       #print("Init")
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.observer = observer
        self.termination = termination

        self.err_count = 0

        self.prevTime = 0

        self.initial_run = True

        ##print("launch torcs")
        self.command = "torcs -nofuel -nodamage -nolaptime"
        if self.vision:
            self.command += " -vision"
        if self.observer:
            self.command += " -observer"

        self.reset_torcs()

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        """
        if throttle is False:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        else:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))

        if vision is False:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
            self.observation_space = spaces.Box(low=low, high=high)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
            self.observation_space = spaces.Box(low=low, high=high)
    def step(self, u, target_speed):
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Automatic Throttle Control by Snakeoil
        vabs = math.sqrt(client.S.d.speedX*client.S.d.speedX + client.S.d.speedY*client.S.d.speedY)
        if self.throttle is False:
            #target_speed = self.default_speed
            if  vabs < target_speed:# - (client.R.d['steer']*50):
                client.R.d['accel'] += .02
            else:
                client.R.d['accel'] -= .02

            if client.R.d['accel'] > 0.4:
                client.R.d['accel'] = 0.4

            #if client.S.d.speedX < 10:
            #    client.R.d['accel'] += 1/(client.S.d.speedX+.1)

            # Traction Control System
            #if vabs > 20 and ((client.S.d.wheelSpinVel[2]+client.S.d.wheelSpinVel[3]) -
            #   (client.S.d.wheelSpinVel[0]+client.S.d.wheelSpinVel[1]) > 5) :
            #    action_torcs['accel'] -= 0.2

        else:
            #use delta accel
            action_torcs['accel'] = this_action['accel']

        #  Automatic Gear Change by Snakeoil
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            if client.S.d.speedX > 50:
                action_torcs['gear'] = 2
            elif client.S.d.speedX > 80:
                action_torcs['gear'] = 3
            elif client.S.d.speedX > 110:
                action_torcs['gear'] = 4
            elif client.S.d.speedX > 140:
                action_torcs['gear'] = 5
            elif client.S.d.speedX > 170:
                action_torcs['gear'] = 6
        # Save the previous full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        server_comm = client.get_servers_input()

        # Get the current full-observation from torcs
        obs = client.S.d

        if not server_comm:
            print "communication failue"
            client.R.d['meta'] = True

        # Make an obsevation from a raw observation vector from TORCS
        self.observation = self.make_observaton(obs)

        # t_delta = obs.curLapTime - self.prevTime
        # print "++++ Lap Time delta", t_delta
        # self.prevTime = obs.curLapTime

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs.track)
        trackPos = np.array(obs.trackPos)
        sp = np.array(obs.speedX)
        damage = np.array(obs.damage)
        rpm = np.array(obs.rpm)
        if u.size == 1:
            vf = 50
        else:
            vf = 10 + sp
        progress = vf * (np.cos(obs.angle) - np.abs(np.sin(obs.angle)) - np.abs(obs.trackPos))
        progress = max(progress, 0.) + indicator(1 - np.abs(obs.trackPos))
        reward = progress
        # if u.size == 1:
        #     print '***** Vx', obs.speedX, "Vy", obs.speedY, "acc", client.R.d['accel'], "tag speed", target_speed
        # else:
        #     print '***** Vx', obs.speedX, "Vy", obs.speedY, "acc", client.R.d['accel']

        # collision detection
        #if obs.damage - obs_pre.damage > 0:
        #    reward = -500

        # Termination judgement #########################
        #if self.termination and track.min() < 0 :
        #    print "Episode is terminated: car is out of track"
        #    reward = - 100
        #    client.R.d['meta'] = True

        #if self.termination and self.terminal_judge_start < self.time_step:
        #    print "Episode terminated: speed of agent is too small"
        #    if progress < self.termination_limit_progress:
        #        client.R.d['meta'] = True

        if self.termination and (np.cos(obs.angle) < 0 or np.abs(np.sin(obs.angle)) > .97):
            print "Episode is terminated: agent runs backward"
            client.R.d['meta'] = True

        # if self.termination and client.S.d.speedX <= 3 and self.time_step >= 80:
        #     print "Episode is terminated: velocity is small"
        #     client.R.d['meta'] = True

        rebound = False
        if self.termination and abs(trackPos) >= 1.:
            if self.err_count >= 150 or client.S.d.speedX < client.S.d.speedX <= 3:
                print "Episode is terminated: trackPos >= 1, err", self.err_count
                client.R.d['meta'] = True
                self.err_count = 0
            else:
                rebound = True
            self.err_count += 1
        else:
            self.err_count = 0

        if client.R.d['meta'] is True: # Send a reset signal
            self.initial_run = False
            client.respond_to_server()
        self.time_step += 1
        return self.get_obs(), reward, client.R.d['meta'], rebound, {}

    def reset(self, relaunch=False):
        #print("Reset")

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        self.observation = self.make_observaton(obs)

        self.last_u = None

        self.initial_reset = False
        return self.get_obs()

    def end(self):
        os.system('pkill torcs')

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
       #print("relaunch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        os.system("%s &" % self.command)
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)

    def agent_to_torcs(self, u):
        torcs_action = {'steer': u[0]}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({'accel': u[1]})

        if self.gear_change is True: # gear change action is enabled
            torcs_action.update({'gear': u[2]})

        return torcs_action

    def obs_vision_to_image_rgb_proto(self, obs_image):
        image_jpeg = np.frombuffer(obs_image, dtype = np.uint8)
        image = cv2.imdecode(image_jpeg, cv2.CV_LOAD_IMAGE_COLOR)
        return image.reshape((SCREEN_HEIGHT, SCREEN_WIDTH, 3))[::-1, :, :]

    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        rgb = []
        # group rgb values together.
        # Format similar to the observation in openai gym

        #print "Total image bytes: %d" % len(image_vec)
        for i in range(0, SCREEN_WIDTH*SCREEN_HEIGHT):
            temp = tuple(image_vec[3*i:3*i+3])
            rgb.append(temp)
       
        return np.array(rgb, dtype=np.uint8).reshape((SCREEN_HEIGHT, SCREEN_WIDTH, 3))[::-1,:,:]

    def make_observaton(self, raw_obs):
        assert self.vision is not False
        if self.vision is False:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ',
                     'opponents',
                     'rpm',
                     'track',
                     'wheelSpinVel']
            Observation = col.namedtuple('Observation', names)
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32))
        elif self.vision:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ', 'angle', 'damage',
                     'opponents',
                     'rpm',
                     'track',
                     'trackPos',
                     'wheelSpinVel',
                     'acceleration',
                     'breaking',
                     'gear',
                     'steering',
                     'lapTime',
                     'img']
            Observation = col.namedtuple('Observation', names)
            image_rgb = self.obs_vision_to_image_rgb_proto(raw_obs.img)

            return Observation(
                focus=np.array(raw_obs.focus, dtype=np.float32)/200.,
                speedX=np.array(raw_obs.speedX, dtype=np.float32)/300.0,
                speedY=np.array(raw_obs.speedY, dtype=np.float32)/300.0,
                speedZ=np.array(raw_obs.speedZ, dtype=np.float32)/300.0,
                angle=np.array(raw_obs.angle, dtype=np.float32)/3.1416,
                damage=np.array(raw_obs.damage, dtype=np.float32),
                opponents=np.array(raw_obs.opponents, dtype=np.float32)/200.,
                rpm=np.array(raw_obs.rpm, dtype=np.float32)/10000,
                track=np.array(raw_obs.track, dtype=np.float32)/200.,
                trackPos=np.array(raw_obs.trackPos, dtype=np.float32)/1.,
                wheelSpinVel=np.array(raw_obs.wheelSpinVel, dtype=np.float32),
                steering=raw_obs.steering,
                gear=int(raw_obs.gear),
                acceleration=raw_obs.acceleration,
                breaking=raw_obs.breaking,
                lapTime = raw_obs.curLapTime,
                img=image_rgb
            )

