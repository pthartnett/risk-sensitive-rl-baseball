# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 21:38:57 2025

@author: pthar
"""

from pybaseball import statcast

# Get Statcast data for a specific date range
data = statcast(start_dt='2024-04-01', end_dt='2025-11-07')

#create booleans for runners on base
data['runner1'] = data['on_1b'].notna()*1
data['runner2'] = data['on_2b'].notna()*1
data['runner3'] = data['on_3b'].notna()*1

data[['zone','plate_x','plate_z']].groupby('zone').agg(['max','min'])

swing_desc = ['foul','foul_pitchout','foul_tip','hit_into_play','swinging_strike','swinging_strike_blocked', 'bunt_foul_tip','foul_bunt','missed_bunt']
bunt_desc = []

data['swing'] = 0
data.loc[data['description'].isin(swing_desc),'swing'] = 1

relevant_fields = ['pitch_name', 
                   'zone',
                   'outs_when_up',
                   'runner1',
                   'runner2',
                   'runner3',
                   'balls',
                   'strikes']

swing_counts = data[relevant_fields+['swing']].groupby(relevant_fields).count()

