# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 16:15:10 2025

@author: pthar
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

def draw_statcast_pitch_zone(train):
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    pitch_table = axes[0]
    
    pitches = sorted(list(set(train.pitch_name)))
    data = [[item] for item in pitches]
    table = pitch_table.table(
        cellText=data,
        colLabels=['Pitch Name'],
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(20)
    table.scale(1, 2.2) 
    
    # MODIFICATION 2: Make column labels bold
    for (i, j), cell in table.get_celld().items():
        if i == 0: # Row 0 is the header row
            cell.set_text_props(fontweight='bold')
            cell.set_text_props(fontsize=24)
            
    
    #pitch_table.set_title("Pitch Types", fontsize=16)
    
    # Hide axes ticks and spines for the table subplot
    pitch_table.axis('off')
    
    ax = axes[1]
    
    SZ_TOP = 1.2
    SZ_BOT = 0.2
    SZ_LEFT = -0.35
    SZ_RIGHT = 0.35
    
    X_MIN = -0.55
    X_MAX = 0.55
    Z_MIN = 0
    Z_MAX = 1.4
    
    sz_v_step = (SZ_TOP - SZ_BOT) / 3 # Vertical step size (approx 0.607 ft)
    sz_h_step = (SZ_RIGHT - SZ_LEFT) / 3 # Horizontal step size (approx 0.472 ft)

    # Define the 3x3 grid lines
    x_lines = [SZ_LEFT, SZ_LEFT + sz_h_step, SZ_LEFT + 2 * sz_h_step, SZ_RIGHT]
    y_lines = [SZ_BOT, SZ_BOT + sz_v_step, SZ_BOT + 2 * sz_v_step, SZ_TOP]
    
    for y in y_lines:
        ax.plot([SZ_LEFT, SZ_RIGHT], [y, y], color='black', linestyle='-', alpha=0.5)
    for x in x_lines:
        ax.plot([x, x], [SZ_BOT, SZ_TOP], color='black', linestyle='-', alpha=0.5)
        
    main_box = patches.Rectangle((SZ_LEFT, SZ_BOT), SZ_RIGHT - SZ_LEFT, SZ_TOP - SZ_BOT, 
                                 fill=False, edgecolor='black', linewidth=3)
    ax.add_patch(main_box)

    zone_number = 1
    for i in range(2, -1, -1): 
        y_center = y_lines[i] + sz_v_step / 2
        for j in range(3): 
            x_center = x_lines[j] + sz_h_step / 2
            ax.text(x_center, y_center, str(zone_number), 
                    color='black', ha='center', va='center', fontsize=20, fontweight='bold')
            zone_number += 1
            
    ax.plot([X_MIN, SZ_LEFT], [Z_MAX, SZ_TOP], color='black', linestyle='-', alpha=0.5)
    ax.plot([SZ_RIGHT, X_MAX], [SZ_TOP, Z_MAX], color='black', linestyle='-', alpha=0.5)
    ax.plot([X_MIN, SZ_LEFT], [Z_MIN, SZ_BOT], color='black', linestyle='-', alpha=0.5)
    ax.plot([SZ_RIGHT, X_MAX], [SZ_BOT, Z_MIN], color='black', linestyle='-', alpha=0.5)
    
    ax.text(0, SZ_TOP + 0.1, "11", color='black', ha='center', va='center', fontsize=20, fontweight='bold')

    ax.text(0, SZ_BOT - 0.1, "12", color='black', ha='center', va='center', fontsize=20, fontweight='bold')
    
    ax.text(SZ_LEFT - 0.1, SZ_BOT + (SZ_TOP - SZ_BOT) / 2, "13", 
            color='black', ha='center', va='center', fontsize=20, fontweight='bold')
    
    ax.text(SZ_RIGHT + 0.1, SZ_BOT + (SZ_TOP - SZ_BOT) / 2, "14", 
            color='black', ha='center', va='center', fontsize=20, fontweight='bold')


    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Z_MIN, Z_MAX)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Statcast Pitch Zones from Behind Plate', fontsize=24, fontweight='bold')
    ax.set_xlabel('Plate X (ft)', fontsize=20)
    ax.set_ylabel('Plate Z (ft)', fontsize=20)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    
    plt.show()

draw_statcast_pitch_zone(train)
