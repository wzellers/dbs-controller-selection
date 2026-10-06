import numpy as np
from openpyxl import Workbook, load_workbook
import os

# Max recorded value of p_beta with the new function is 2.261527e+05
# Healthy 1.289175e+05
# Min recorded value of p_beta (150, 400) -> min 4.154213e+04, max 1.071793e+05

def calculate_pb_new(vgi, Fs=100000, beta_band=(13, 35)):
    """
    Calculate beta band magnitude using FFT (absolute values).
    
    vgi: 1D numpy array (signal)
    Fs: sampling frequency
    beta_band: tuple (low, high) Hz
    """
    N = len(vgi)
    fft_vals = np.fft.fft(vgi)
    fft_freqs = np.fft.fftfreq(N, 1/Fs)
    # Take absolute magnitude
    fft_magnitude = np.abs(fft_vals)
    # Select beta band
    idx = np.logical_and(fft_freqs >= beta_band[0], fft_freqs <= beta_band[1])
    # Sum absolute magnitudes in beta band
    beta_abs_sum = np.sum(fft_magnitude[idx])
    
    return beta_abs_sum

def calculate_pb_absolute(vgi):
    super_signal = np.mean(vgi, axis=0)
    total = calculate_pb_new(super_signal[:95000],100000)
    return total


def normalize_pb_absolute(pb):
    min = 33178
    max = 234149
    return float((pb - min)/(max - min))

def normalize_pb(pb):
    min = 100000
    max = 1120000
    return float((pb - min)/(max - min))

def normalize_rms(rms):
    min = 0 
    # min = 64.8 # for 55,500
    max = 1161.9
    return float((rms - min)/(max - min))

# def normalize_misses(misses):
#     min = 18920
#     max = 20000
#     return float((misses - min)/(max - min))

# def normalize_misses_1000(misses):
#     min = 94601
#     max = 25000
#     return float((misses - min)/(max - min))
    
# def normalize_consecutive_misses_1000(misses):
#     min = 526 # for freq = 190
#     max = 9970 # for freq = 10
#     return float((misses - min)/(max - min))

def normalize_consecutive_misses(misses):
    # min = 558 # for freq = 170
    min = 526 # for freq = 180
    # max = 1970 # for freq = 50
    # max = 1788 # for freq = 55
    max = 100020 # for freq = 0
    return float((misses - min)/(max - min))

def calculate_pb(vgi):
    pb_sum = 0
    for i in range(len(vgi)):
        pb = sum(abs(np.fft.fft(vgi[i][300:]))[13:35])
        pb_sum += pb
    return pb_sum/10


def compute_rms(I):
    I = np.array(I, dtype=np.float64)
    integral_approx = np.sum(I[0]**2)
    IR_MS = np.sqrt(integral_approx / len(I[0]))
    return IR_MS

def count_zeros(tmax, freq, curr):
    dt = 0.01
    t = np.arange(0, tmax + dt, dt)
    Idbs = np.zeros(len(t))
    iD = curr
    pulse = iD * np.ones(int(0.3 / dt))
    i = 0
    while i < len(t):
        if(i + len(pulse) > len(Idbs)):
            Idbs[i : len(Idbs)] = pulse[:len(Idbs)-i]
        else:
            Idbs[i : i + len(pulse)] = pulse
        instfreq = freq
        isi = 1000 / instfreq
        i += round(isi / dt)
    zero_count = 0
    for num in Idbs:
        if num == 0:
            zero_count += 1
    return zero_count

def count_max_consecutive_zeros(tmax, freq, curr):
    dt = 0.01
    t = np.arange(0, tmax + dt, dt)
    Idbs = np.zeros(len(t))
    iD = curr
    pulse = iD * np.ones(int(0.3 / dt))
    i = 0
    while i < len(t):
        if(i + len(pulse) > len(Idbs)):
            Idbs[i : len(Idbs)] = pulse[:len(Idbs)-i]
        else:
            Idbs[i : i + len(pulse)] = pulse
        instfreq = freq
        isi = 1000 / instfreq
        i += round(isi / dt)
    max_zeros = 0
    current_zeros = 0

    for num in Idbs:
        if num == 0:
            current_zeros += 1
            max_zeros = max(max_zeros, current_zeros)
        else:
            current_zeros = 0

    return max_zeros

def append_to_excel(file_path, sheet_name, data):
    """
    Appends a list of values (`data`) to the next empty row of an Excel sheet.

    Parameters:
    - file_path: str, path to the Excel file
    - sheet_name: str, name of the sheet to write to
    - data: list, values to write to the next row
    """
    if os.path.exists(file_path):
        wb = load_workbook(file_path)
        if sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
        else:
            ws = wb.create_sheet(sheet_name)
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = sheet_name

    # Find the next empty row
    next_row = ws.max_row + 1 if any(cell.value is not None for cell in ws[ws.max_row]) else ws.max_row

    # Write the data
    for col, value in enumerate(data, start=1):
        ws.cell(row=next_row, column=col, value=value)

    # Save the file
    wb.save(file_path)


