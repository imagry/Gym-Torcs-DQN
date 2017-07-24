#### [Blog post with details and description.](http://www.imagry.co/?p=49609)

# Installation:

0. Install Caffe with opencv2 (opencv is also neeeded for this version of gym-torcs)
    https://github.com/BVLC/caffe

1. install Torcs dependencies
    1.0 remove old gym-torc if necessary: 
    * make distclean
    
    1.1 xautomation (http://linux.die.net/man/7/xautomation)
    * apt-get install xautomation
    
    1.2 OpenAI-Gym (https://github.com/openai/gym)
    
    * apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
    * pip install gym

2. Install gym-torcs
    2.1 install vtorcs-RL-color dependencies
    * sudo apt-get install libglib2.0-dev  libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev libplib-dev  libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev libxrandr-dev libpng12-dev 

    2.2 install gym-torcs
    
    * cd /gym-torcs/
command:
    * SRC_DIR=proto DST_DIR=. CPP_DST_DIR=vtorcs-RL-color/src/drivers/scr_server/; protoc -I=$SRC_DIR --cpp_out=$CPP_DST_DIR --python_out=$DST_DIR $SRC_DIR/car_state.proto          

    2.2.1 install vtorcs-RL-color

    * cd /gym-torcs/vtorcs-RL-color
    * ./configure
    * make -j8
    * if "no ossspec.h" or "no tgfclien.h" errors just restart make
    * make install
    * make datainstall

    2.3 Initialize race
    * sudo torcs
        Configure driver:
        Race --> Quick Race --> Configure Race-->(Select track)Accept
        -->On the left column Deselect "Damned" driver

        Quit all menues

        To temporary increase resolution if font is too small
        before configuring driver 
        Option-->Display-->choose resolution 
        and affter configuring driver
        Option-->Display-->choose Screen resolution 256x192 
        
3. export Caffe and CUDA paths
* export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64:/usr/local/lib
* export PYTHONPATH=/home/sergey/repo/caffe/python:$PYTHONPATH
4. Download models [from this link](https://imagry-my.sharepoint.com/personal/sergey_imagry_co/_layouts/15/guestaccess.aspx?folderid=0042e1f06f8074b83b02ac8862ade0673&authkey=AZt1NkRuRmtTh4wDz4ckso0) and put them into working directory specified in current_dir in dqn.py 
- r18nb.caffemodel is original resnet-18 model trained on the imagenet. Batch normalization layers merged into convolutions
- q_solver.caffemodel is dqn current training model
- qq_target.caffemodel is dqn taret model, running avergae of training model

# Model archiecture:

Protofiles for models are in the directory resnet_torcs
- critic_batch_dqn.prototxt for qq_target.caffemodel
- dqn_critic_solver.prototxt for q_solver.caffemodel  
- critic_deploy_dqn.prototxt is protofile for driving agent
- resnet18_nbn.prototxt is not used in project, it's prtofile for resnet-18 with merged BN


# Training:

For general description check [this blog post](http://www.imagry.co/?p=49609)
How to start:

- in dqn.py you should modify:
  - you current directory, this is the directory where your initial models are
    in the line 36:
    current_dir = '/home/sergey/repo/gym-torcs/'
  - chose replay or training in the line 35, False for training
    play = False 
   - chose to start form imagenet renet-18 model (r18nb.caffemodel), line 509
    first_run = True
   - or from saved dqn network
    first_run = False
   - delete  and reinitialzie replay buffer, line 510:
    reset_buffer = True
   - save model and replay buffer in nnn iterations, line 533
    save_in_iters = nnn
   - max number of steps per track, line 532 (each step take at least 0.4 second)
    max_steps = 5000

- run python dqn.py

- Important parameters
    - size of replay buffer, line 21
        BUFFER_SIZE = 70000
    - gamma and tau, line 514,515 (explained in blog post)
        GAMMA =0.99
        TAU = .0001
    - steering action amplitude, line 26 (require retraining):
        DELTA_A = .85
    - exploration noise, line 518-520
        initial value of noise (for starting training recommended 0.5, for continuation of training recommended 0.05)
        act_noise_init = .5
        minimal value of noise
        act_noise_final = .01
        inverse scale factor
        act_noise_interval = 100000
    - action smoother and limiter, line 528-529 (explained in blog post)
        action_smoother = .33
        action_limiter = .33
    - avergae speed and speed variance, line 560-563
        c_speed = 35
        delta_speed = 5 
- Useful parameters
    - switch speed randomly after swith_count_min(1 + uniform(0,1)) steps
        swith_count_min = 50
    - n-step dqn (see blog post) with n =  max_n_batches*BATCH_SIZE.
        max_n_batches = 16
    - n_step ratio - use n-step with n_step_ratio probaility and normal dqn for the rest.  For normal dqn use n_step_ratio = 0
        n_step_ratio = 0.75
    - priority sampling with k-try (see blog post)
        k_priority_try = 2
    - stop episod after max_errors
        max_errors = 50
- Risky parameters
    - Q function spatial TV-L2 regularization, may make action smoother. Should be very small.
        lambda_spatial_q = 0
    - parameter for TD-Lambda line 437  (see blog post), lesser alpha is only for very small exploration noise
        alpha = 1

 #### **** Have fun! ****

Acknowledgement:
[Caffe](https://github.com/BVLC/caffe) is developed by Berkeley AI Research (BAIR)/The Berkeley Vision and Learning Center (BVLC) and community contributors
Naoto Yoshida is the author of the [gym torcs](https://github.com/ugo-nama-kun/gym_torcs).
Implementation of replay buffer is based on [Ben Lau](https://yanpanlau.github.io) work.

    



