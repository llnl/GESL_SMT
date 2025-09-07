import pandas as pd
import numpy as np

def positive_seq_conversion(three_phase_meas_polar, device):
    a = np.exp(1j * 2 * np.pi / 3)
    three_phase_meas_polar.columns = [col.split('.')[-1] for col in three_phase_meas_polar.columns]
    df_current = three_phase_meas_polar[[col for col in three_phase_meas_polar.columns if any(s in col for s in 'i')]]
    df_voltage = three_phase_meas_polar[[col for col in three_phase_meas_polar.columns if any(s in col for s in 'v')]]
    df_current_complex = pd.DataFrame()
    df_voltage_complex = pd.DataFrame()
    df_pos_seq = pd.DataFrame()
    for phase in ['a','b','c']:
        current_complex = polar_to_complex(df_current['i' + phase + '_m'], df_current['i' + phase + '_a'])
        voltage_complex = polar_to_complex(df_voltage['v' + phase + '_m'], df_voltage['v' + phase + '_a'])
        df_current_complex['i' + phase] = current_complex
        df_voltage_complex['v' + phase] = voltage_complex
    pos_current = (df_current_complex['ia'] + a * df_current_complex['ib'] + a**2 * df_current_complex['ic']) / 3
    pos_voltage = (df_voltage_complex['va'] + a * df_voltage_complex['vb'] + a**2 * df_voltage_complex['vc']) / 3
    pos_seq_values = pd.concat([pos_current, pos_voltage], axis=1)
    pos_seq_values.columns=['i','v']
    df_pos_seq[device + '.ip_a'] = np.degrees(np.angle(pos_seq_values['i']))
    df_pos_seq[device + '.ip_m'] = np.abs(pos_seq_values['i'])
    df_pos_seq[device + '.vp_a'] = np.degrees(np.angle(pos_seq_values['v']))
    df_pos_seq[device + '.vp_m'] = np.abs(pos_seq_values['v'])
    return df_pos_seq

def polar_to_complex(mag, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    return mag * np.exp(1j * angle_rad)

def mg_data_pre_process(input_file,string_list):
    three_phase_meas = ['ia_m','ib_m','ic_m',
                        'ia_a','ib_a','ic_a',
                        'va_m','vb_m','vc_m',
                        'va_a','vb_a','vc_a']
    df = pd.read_csv(input_file)
    df_three_phase = df[[col for col in df.columns if any(s in col for s in string_list)]]
    devices_list = list(set([s.split('.')[0] for s in df_three_phase.columns]))
    df_output = pd.DataFrame()
    for device in devices_list:
        df_device = df_three_phase[[col for col in df_three_phase.columns if any(s in col for s in device)]]
        df_device = df_device[[col for col in df_device.columns if any(s in col for s in three_phase_meas)]]
        pos_seq_values = positive_seq_conversion(df_device, device)
        pos_seq_values.insert(0, device + '.f', df_three_phase[device + '.f'].copy())
        pos_seq_values.insert(0, device + '.df', df_three_phase[device + '.df'].copy())
        df_output = pd.concat([df_output, pos_seq_values], axis=1)
    df_output.insert(0, 'Time', df['Time'].copy())  
    return df_output, devices_list


    
    
  