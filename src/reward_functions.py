# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 14:21:41 2025

@author: pthar
"""

#baseline run    
def calc_pitch_reward(pitch, runs_scored, old_state):
    run_reward = -6
    out_reward = 1
    event_reward ={
        'ball':-.01,
        'catcher_interf':0,
        'double':-0.2,
        'double_play':2*out_reward,
        'error':0,
        'foul':0.01,
        'hit_by_pitch':-0.1,
        'home_run':0,
        'out':out_reward,
        'sacrifice':out_reward - 0.1,
        'single':-0.1,
        'strike':0.01,
        'triple':-0.3,
        'triple_play':3*out_reward
        }
    if old_state.balls==3:
        event_reward['ball'] = -0.1
    if old_state.strikes==2:
        event_reward['strike'] = out_reward
        event_reward['foul'] = 0
    
    return run_reward*runs_scored + event_reward[pitch['pitch_outcome']]



#outs only
def calc_pitch_reward(pitch, runs_scored, old_state):
    run_reward = 0
    out_reward = 1
    event_reward ={
        'ball': 0.0,
        'catcher_interf':0,
        'double': 0.0,
        'double_play':2*out_reward,
        'error':0,
        'foul':0.0,
        'hit_by_pitch': 0,
        'home_run':0,
        'out':out_reward,
        'sacrifice': out_reward,
        'single': 0.0,
        'strike': 0.0,
        'triple': 0,
        'triple_play':3*out_reward
        }
    if old_state.balls==3:
        event_reward['ball'] = 0.0
    if old_state.strikes==2:
        event_reward['strike'] = out_reward
        event_reward['foul'] = 0
    
    return run_reward*runs_scored + event_reward[pitch['pitch_outcome']]

#runs only
def calc_pitch_reward(pitch, runs_scored, old_state):
    run_reward = -6
    out_reward = 0
    event_reward ={
        'ball': 0.0,
        'catcher_interf':0,
        'double': 0.0,
        'double_play':2*out_reward,
        'error':0,
        'foul':0.0,
        'hit_by_pitch': 0,
        'home_run':0,
        'out':out_reward,
        'sacrifice': out_reward,
        'single': 0.0,
        'strike': 0.0,
        'triple': 0,
        'triple_play':3*out_reward
        }
    if old_state.balls==3:
        event_reward['ball'] = 0.0
    if old_state.strikes==2:
        event_reward['strike'] = out_reward
        event_reward['foul'] = 0
    
    return run_reward*runs_scored + event_reward[pitch['pitch_outcome']]

#limit bad innings
def calc_pitch_reward(pitch, runs_scored, old_state):
    run_reward = -1
    out_reward = 1
    event_reward ={
        'ball':-.01,
        'catcher_interf':0,
        'double':-0.2,
        'double_play':2*out_reward,
        'error':0,
        'foul':0.01,
        'hit_by_pitch':-0.1,
        'home_run':-0.1,
        'out':out_reward,
        'sacrifice':out_reward - 0.1,
        'single':-0.1,
        'strike':0.01,
        'triple':-0.3,
        'triple_play':3*out_reward
        }
    
    if old_state.strikes==2:
        event_reward['strike'] = out_reward
        event_reward['foul'] = 0
        
    less_out_multiplier = (2-old_state.outs_when_up)*.1+1
    on_base_multiplier = 1+(old_state.runner1*.1+old_state.runner2*.2+old_state.runner3*.3)/2.0
    
    if old_state.balls==3:
        event_reward['ball'] = -0.1*less_out_multiplier*on_base_multiplier
    event_reward['hit_by_pitch'] = event_reward['hit_by_pitch']*less_out_multiplier*on_base_multiplier
    event_reward['single'] = event_reward['single']*less_out_multiplier*on_base_multiplier
    event_reward['double'] = event_reward['double']*less_out_multiplier*on_base_multiplier
    event_reward['triple'] = event_reward['triple']*less_out_multiplier*on_base_multiplier
    event_reward['home_run'] = event_reward['home_run']*less_out_multiplier*on_base_multiplier
    
    total_run_reward = run_reward*(runs_scored*less_out_multiplier*on_base_multiplier)**2
    enhanced_reward = total_run_reward + event_reward[pitch['pitch_outcome']]
    
    return enhanced_reward