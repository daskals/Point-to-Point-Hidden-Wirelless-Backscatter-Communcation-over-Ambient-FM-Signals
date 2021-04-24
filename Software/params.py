#######################################################
#     Spiros Daskalakis                               #
#     Last Revision: 7/8/2020                         #
#     Python Version:  3.8.4                          #
#     Email: daskalakispiros@gmail.com                #
#     Website: www.daskalakispiros.com                #
#######################################################
import numpy as np

# Set the sample Rate of MyDaq
daq_fs = 200e3  # 200 kS/s maximum
daq_voltage_scale = 1.5  # volts

# Set the sample Rate of Sound Card
sound_card_fs = 192000  # 48 kS/s maximum
sound_card_resolution = 24  # bit

# tag num of packets
Num_of_packets = 1
#######################################################
# Mode parameters
# Mode 2: MY_DAQ
# Mode 1: Sound_Card
# Mode 0: virtual cir
mode_tx_generator = 0
#######################################################
# Set the sample Rate of Tag
if mode_tx_generator == 2 or mode_tx_generator == 0:
    Fs_tx = daq_fs
elif mode_tx_generator == 1:
    Fs_tx = sound_card_fs
#######################################################
# Set the sample Rate of smartphone (receiver)
Fs_rx = 48e3  # 48 kS/s maximum
#######################################################
# Our signal is going to be a chirp frequency signal
LEFT = 15200  # Chirp start frequency 15.2 kHz
RIGHT = 20000  # Chirp end frequency 23.2 kHZ
# Chirp BW
BW = RIGHT - LEFT
#######################################################
# Spreading factor 8 or 10  or 12
SF = 10
#######################################################
# Coding rate CR=0...4 for rate 4/(4+CR)
CR = 4
#######################################################
# Symbol period: 2^SF/BW
T_symbol = (2 ** SF) / BW
#######################################################
# Symbol rate: 1/(2^SF/BW)
Symbol_rate = 1 / T_symbol
# Bit rate: SF*((4/(4+CR))/(2^SF/BW))
Bit_rate = SF * (1 / T_symbol) * (4 / (4 + CR))
#######################################################
chirp_num_samples_rx = Fs_rx * (2 ** SF) / BW
chirp_num_samples_tx = Fs_tx * (2 ** SF) / BW
#######################################################
# Mode parameters
# Mode 1: raw_symbols_mode
# Mode 2: gray_bits_mode
# Mode 3: coded_bits_mode
mode_tx_coding = 1
#######################################################
if mode_tx_coding == 2 or mode_tx_coding == 1:
    CR = 0
#######################################################
# Chirp type
# 1=sin wave symbol
# 2=square wave symbol
# 3=complex symbol
mode_tx_chirp = 1
#######################################################
preamble_tx = np.array([-1, -1, -1, -1, -1, 1, 1, 1])
raw_symbols_tx = np.array([20, 43, 230, 250, 600, 364,4])

#######################################################
# set tx_fig_en to '1' to enable the figures
Figs_tx_en = 1
Figs_rx_en = 0
######################################################
CFO_corr_en = 0
########################################################
