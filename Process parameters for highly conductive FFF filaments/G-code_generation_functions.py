# -*- coding: utf-8 -*-
"""
G_code_functions.py

Includes functions for G_code generation for FDM 3D printing.
"""

import numpy as np


def E_calc(extrude_length, print_params, dim_params):
    """
    Function calculates length of filament [mm] needed for the printing of one trace with specified dimensions.
    
    input:
        - print_params = [flow_rate [/], D_nozzle [mm], D_filament [mm], print_speed [mm/s], move_speed [mm/s]]:
        - dim_params = [w, y_s, l_x, h]:
            - w       ... trace width [mm]
            - y_s     ... trace spacing [mm]
            - l_x     ... trace length [mm]
            - h       ... layer height [mm]
    output:
        - E           ... length of filament to be extruded
    """
    
    flow_rate = print_params[0]
    D_nozzle = print_params[1]
    D_filament = print_params[2]
    
    h = dim_params[-1]
    w = dim_params[0]
    
    A_in = np.pi * D_filament**2 / 4
    A_out = (w - h) * h + np.pi * h**2 / 4
    
    E = A_out / A_in * extrude_length * flow_rate
    
    return E


def G1_gen(zero_point, line_num, layer_num, print_params, dim_params, sample_name):
    """
    Generates G-code for a sample printed with lines in string format.
    
    input:
        - zero_point  ... sample zero point for nozzle tip = (x0, y0, z0) [mm]
        - line_num    ... number of lines
        - layer_num   ... number of layers
        - print_params =
            flow_rate [/],
            D_nozzle [mm],
            D_filament [mm],
            print_speed [mm/s],
            move_speed [mm/s],
            retract_distance [mm],
            retract_speed [mm/s],
            nozzle_lift [mm],
            wipe_distance [mm],
            wipe_speed [mm/s],
        - dim_params = [w, y_s, l_x, h]:
            - w       ... trace width [mm]
            - y_s     ... trace spacing [mm]
            - l_x     ... trace length [mm]
            - h       ... layer height [mm]
    output:
        - G_code    ... G-code string
    """
    
    # Unpack params:
    w = dim_params[0]
    l_x = dim_params[2]
    y_s = dim_params[1]
    h = dim_params[3]
    retract_distance = print_params[5]
    retract_speed = print_params[6]
    nozzle_lift = print_params[7]
    wipe_distance = print_params[8]
    wipe_speed = print_params[9]
    flow_rate = print_params[0]
    print_speed = print_params[3]
    move_speed = print_params[4]
    
    # Set feedrate:
    print_feedrate = print_speed * 60 # [mm/min]
    move_feedrate = move_speed * 60 # [mm/min]
    retract_feedrate = retract_speed * 60 # [mm/min]
    wipe_feedrate = wipe_speed * 60 # [mm/min]
    
    # Start of G-code:
    G_code = ';' + 60*'=' + '\n'
    G_code += f'; Start of python generated G-code for {sample_name}\n\n'
    G_code += '; Printing Parameters:\n'
    G_code += f'; \t flowrate = {flow_rate*100:.1f} %\n'
    G_code += f'; \t print_speed = {print_speed} mm/s\n'
    G_code += f'; \t move_speed = {move_speed} mm/s\n'
    G_code += '; Sample Dimension Parameters:\n'
    G_code += f'; \t line width = {w} mm\n'
    G_code += f'; \t Y-axis line spacing = {y_s} mm\n'
    G_code += f'; \t X_axis line length = {l_x} mm\n'
    G_code += f'; \t layer height = {h} mm\n\n'
    
    G_code += f'; Zero Point = {zero_point} mm\n'
    G_code += f'; Number of lines = {line_num} \n'
    G_code += f'; Number of layers = {layer_num} \n\n'
    
    start_point = zero_point
    for i in range(layer_num): # loop for layers
        
        # Layer comment and description:
        G_code += ';' + 60*'-' + '\n'
        G_code += f'; Layer {i+1}, z = {start_point[2]}:\n\n'
        
        # Set moving feedrate:
        G_code += f'G1 F{move_feedrate:.5f} ; setting move feedrate\n'
        # Move to layer start point:
        G_code += f'G1 X{start_point[0]:.5f} Y{start_point[1]:.5f} Z{start_point[2] + nozzle_lift:.5f} ; moving to layer start position\n\n'
        
        for j in range(line_num): # loop for lines 
            G_code += f'; Line {j+1}:\n'
            # Set moving feedrate:
            G_code += f'G1 F{move_feedrate:.5f} ; setting move feedrate\n'
            # starting point:
            G_code += f'G1 X{start_point[0]:.5f} Y{start_point[1]:.5f} Z{start_point[2]:.5f} ; moving to first point of the line\n'
            G_code += f'G1 E{retract_distance:.5f} ; unretract\n'
                
            # extrude one line:
            G_code += f'G1 F{print_feedrate:.5f} ; setting print feedrate\n'
            E_val = E_calc(l_x, print_params, dim_params)
            G_code += f'G1 X{start_point[0] + l_x:.5f} Y{start_point[1]:.5f} E{E_val:.5f} ; printing line {j+1}\n' 
            # retract, wipe and raise:
            G_code += f'G1 F{retract_feedrate} ; set feedrate for retract and wipe\n'
            G_code += f'G1 X{start_point[0] + l_x - wipe_distance:.5f} Y{start_point[1]:.5f} E{-retract_distance:.5f} ; retract and wipe\n'
            G_code += f'G1 Z{start_point[2] + nozzle_lift:.5f} ; setting Z-axis height for moving\n'

            # override start position    
            start_point = [start_point[0] + 0, start_point[1] + y_s, start_point[2]]

            G_code += '\n'
        
        # before layer change:
        G_code += '; Before layer change:\n'
        G_code += 'G92 E0.0\n\n'
        
        # override start point:
        start_point = [zero_point[0], zero_point[1], start_point[2] + h]
    
    # End retract
#    G_code += f'G1 E{-retract_distance:.5f} ; finish retract\n\n'
    
    # End of G-code
    G_code += '; End of python generated G-code\n'
    G_code += ';' + 60*'=' + '\n'
    
    return G_code


def Move_G_code(g_code_to_move, dX=0, dY=0, dZ=0):
    """
    Moves G-code for specified dx, dy, dz in mm.
    
    input:
        g_code_to_move ... g_code in list of string lines format (one continuous line)
        dx, dy, dz ... [mm]
        
    returns:
        moved_g_code_string ... moved g_code in string format (one continuous line)
    """
    
    moved_g_code = []
    
    for line_number, line in enumerate(g_code_to_move):
        
        words = line.split()
        
        if line[0] == ';': # line is a comment
            moved_g_code.append(line)
            continue

        if len(words) == 0: # empty line
            moved_g_code.append(line)
            continue
        
        if words[0] == 'G1':
            
            X_found, Y_found, Z_found = False, False, False
            
            new_line = [words[0]]
            
            for word in words[1:]:
                
                try:
                    if word[0] == 'X' and X_found == False and Y_found == False and Z_found == False:
                        X_found = True
                        
                        old_X = float(word[1:])
                        new_X = old_X + dX

                        new_word = 'X' + str(new_X)
                        new_line.append(new_word)

                    elif word[0] == 'Y' and Y_found == False and Z_found == False:
                        Y_found = True

                        old_Y = float(word[1:])
                        new_Y = old_Y + dY

                        new_word = 'Y' + str(new_Y)
                        new_line.append(new_word)

                    elif word[0] == 'Z' and Z_found == False:
                        Z_found = True

                        old_Z = float(word[1:])
                        new_Z = old_Z + dZ

                        new_word = 'Z' + str(new_Z)
                        new_line.append(new_word)

                    else:

                        new_line.append(word)
                    
                except:
                    print(f'Error at word: {word} at line: {line_number}')
            
            new_line_string = ' '.join(new_line)
            
            if new_line_string.find('\n') == -1:
                new_line_string += '\n'
            
            moved_g_code.append(new_line_string)
            
        else:
            moved_g_code.append(line)
            continue
    
    moved_g_code_string = ''.join(moved_g_code)
    
    return moved_g_code_string

