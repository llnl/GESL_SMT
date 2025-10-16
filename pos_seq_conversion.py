"""
==============================================================================
Three-Phase Measurements to Positive Sequence Conversion Script
==============================================================================

Overview:
    This script processes three-phase electrical measurements (currents and 
    voltages) from a CSV file and computes the positive sequence components 
    for each set of measurements. Positive sequence values are commonly used 
    in power system analysis for fault detection, system monitoring, and control.

Input Requirements:
    - CSV File Structure:
        The input CSV file must contain the following columns for each phase 
        (a, b, c), with magnitudes ('_m') and angles in degrees ('_a'):
            * ia_m, ib_m, ic_m    # Current magnitudes
            * ia_a, ib_a, ic_a    # Current angles (degrees)
            * va_m, vb_m, vc_m    # Voltage magnitudes
            * va_a, vb_a, vc_a    # Voltage angles (degrees)
    - Column Names:
        Column names must match exactly as shown above.
        The order of headers in the CSV file does NOT matter, as long as all required columns are present.
        The script will exit with a warning if any required columns are missing.

Usage:
    Run the script from the command line, specifying your input CSV file path:
        python pos_seq_conversion.py /path/to/your_measurement_file.csv

    The script will:
        - Read three-phase measurements from the specified CSV.
        - Convert polar values (magnitude and angle) to complex numbers.
        - Calculate positive sequence components for current and voltage.
        - Output a DataFrame with:
            * ip_a: Positive sequence current angle (degrees)
            * ip_m: Positive sequence current magnitude
            * vp_a: Positive sequence voltage angle (degrees)
            * vp_m: Positive sequence voltage magnitude
        - Save the results to a file named 'positive_sequence.csv' in the current directory.

Important Notes:
    - All input angles must be in degrees.
    - Ensure all magnitudes are in per unit (p.u.) 
    - The script requires the 'pandas' and 'numpy' Python packages.
    - The script will stop with a warning if required columns are missing.
==============================================================================
"""

import pandas as pd
import numpy as np
import argparse
import sys

def polar_to_complex(magnitude, angle_deg):
    """
    Convert magnitude and angle (in degrees) to a complex number.
    """
    angle_rad = np.deg2rad(angle_deg)
    return magnitude * np.exp(1j * angle_rad)

def positive_seq_conversion(three_phase_meas_polar):
    """
    Calculate the positive sequence components for current and voltage
    from three-phase polar measurements for a single device.

    Parameters:
        three_phase_meas_polar (pd.DataFrame): DataFrame with columns for each phase's magnitude and angle.
        device (str): Device name (used for output column names).

    Returns:
        pd.DataFrame: DataFrame with positive sequence magnitude and angle for current and voltage.
    """
    # Define the 120-degree rotation operator for symmetrical components
    a = np.exp(1j * 2 * np.pi / 3)
    
    # Simplify column names (keep only the suffix, e.g., 'ia_m')
    three_phase_meas_polar = three_phase_meas_polar.copy()
    three_phase_meas_polar.columns = [col.split('.')[-1] for col in three_phase_meas_polar.columns]

    # Extract current and voltage columns
    df_current = three_phase_meas_polar[[col for col in three_phase_meas_polar.columns if col.startswith('i')]]
    df_voltage = three_phase_meas_polar[[col for col in three_phase_meas_polar.columns if col.startswith('v')]]

    # Convert polar to complex for each phase
    df_current_complex = pd.DataFrame()
    df_voltage_complex = pd.DataFrame()
    for phase in ['a', 'b', 'c']:
        df_current_complex[f'i{phase}'] = polar_to_complex(
            df_current[f'i{phase}_m'], df_current[f'i{phase}_a']
        )
        df_voltage_complex[f'v{phase}'] = polar_to_complex(
            df_voltage[f'v{phase}_m'], df_voltage[f'v{phase}_a']
        )

    # Calculate positive sequence for current and voltage
    pos_current = (df_current_complex['ia'] + a * df_current_complex['ib'] + a**2 * df_current_complex['ic']) / 3
    pos_voltage = (df_voltage_complex['va'] + a * df_voltage_complex['vb'] + a**2 * df_voltage_complex['vc']) / 3

    # Prepare output DataFrame
    df_pos_seq = pd.DataFrame({
        f'ip_a': np.degrees(np.angle(pos_current)),
        f'ip_m': np.abs(pos_current),
        f'vp_a': np.degrees(np.angle(pos_voltage)),
        f'vp_m': np.abs(pos_voltage),
    })
    return df_pos_seq

def data_pre_process(input_file):
    """
    Process microgrid data file to extract positive sequence components for all devices.
    Parameters:
        input_file (str): Path to the CSV file containing measurements.
    Returns:
        tuple: (output DataFrame, list of device names)
    """
    # Define expected three-phase measurement columns
    three_phase_meas = [
        'ia_m', 'ib_m', 'ic_m',
        'ia_a', 'ib_a', 'ic_a',
        'va_m', 'vb_m', 'vc_m',
        'va_a', 'vb_a', 'vc_a'
    ]

    # Load data
    df_three_phase = pd.read_csv(input_file)

    # Skip device if not all required columns are present
    if set(three_phase_meas) - {col.split('.')[-1] for col in df_three_phase.columns}:
        print(f"Warning:  Missing some required columns. Exit.")
        exit(1)

    # Compute positive sequence values
    df_pos_seq_values = positive_seq_conversion(df_three_phase)
    return df_pos_seq_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute positive sequence components from three-phase measurement CSV file."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input CSV file containing three-phase measurements."
    )

    args = parser.parse_args()

    try:
        df_pos_seq = data_pre_process(args.input_file)
        df_pos_seq.to_csv('positive_sequence.csv', index=False)
        print(df_pos_seq)
        print("\nPositive sequence results have been saved to 'positive_sequence.csv'.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)