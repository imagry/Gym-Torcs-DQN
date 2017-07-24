#!/bin/bash

# Practice mode => one car on the track
#xte 'key Return'
#xte 'usleep 100000'
#xte 'key Return'
#xte 'usleep 100000'
#xte 'key Up'
#xte 'usleep 100000'
#xte 'key Up'
#xte 'usleep 100000'
#xte 'key Return'
#xte 'usleep 100000'
#xte 'key Return'

# Quick race mode => one car on the track
xte 'usleep 100000'
xte 'key Return'
xte 'usleep 100000'
xte 'key Return'
xte 'usleep 100000'
xte 'key Return'
xte 'usleep 100000'
xte 'key Return'
xte 'usleep 100000'

# Change view to "1st person"
xte 'usleep 100000'
xte 'key F2'
xte 'usleep 10000'
xte 'key F2'
xte 'usleep 10000'
xte 'key F2'
xte 'usleep 10000'
xte 'key F2'
xte 'usleep 10000'
xte 'key F2'
xte 'usleep 10000'
xte 'key F2'
xte 'usleep 10000'
xte 'key F2'
xte 'usleep 10000'
xte 'key F2'
xte 'usleep 10000'
xte 'key F2'
xte 'usleep 10000'
xte 'key F2'

# No cross
xte 'usleep 10000'
xte 'key 4'

# x2 speed for faster simulation
#xte 'usleep 10000'
#xte 'keydown Shift_L'; xte 'key plus'; xte 'keyup Shift_L';


# 0.125 speed for visual smoothiness / no frame skipping
xte 'usleep 10000'
xte 'key minus'
xte 'usleep 10000'
xte 'key minus'
#xte 'usleep 10000'
#xte 'key minus'
