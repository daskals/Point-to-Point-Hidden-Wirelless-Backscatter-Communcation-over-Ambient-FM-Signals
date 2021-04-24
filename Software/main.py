#######################################################
#     Spiros Daskalakis                               #
#     last Revision: 19/03/2021                       #
#     Python Version:  3.9                            #
#     Email: daskalakispiros@gmail.com                #
#     Website: www.daskalakispiros.com                #
#######################################################
from matplotlib import pyplot as plt
from TX import TXtag
from RX import RX_receiver
import numpy as np
import yaml
from box import Box



if __name__ == '__main__':
    mode = 1

    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

    CR = cfg.CR
    SF = cfg.SF

    CHIRP_F_START = cfg.CHIRP_F_START
    CHIRP_F_STOP = cfg.CHIRP_F_STOP
    TX_SAMPLING_RATE = cfg.TX_SAMPLING_RATE
    SOUND_RES = cfg.SOUND_RES
    PREAMBLE = cfg.PREAMBLE
    BW = CHIRP_F_STOP-CHIRP_F_START

    tag = TXtag(mode=mode, CR=CR, SF=SF, BW=BW, CHIRP_F_START=CHIRP_F_START, TX_SAMPLING_RATE=TX_SAMPLING_RATE,
                SOUND_RES=SOUND_RES, PREAMBLE=PREAMBLE)

    header_symbols, symbols, tx_symbols = tag.create_symbols('Spiros')
    packet = tag.create_packet(header_symbols, symbols)

    rf_samples = packet
    #####################################################################################

    rf_signal = rf_samples[:, 0] + 1j * rf_samples[:, 1]

    Zo = np.zeros(15000, dtype=complex)
    rf_signal = np.concatenate((Zo, rf_signal), axis=0)

    rx = RX_receiver(mode=1, CR=CR, SF=SF, BW=BW, CHIRP_F_START=CHIRP_F_START, RX_SAMPLING_RATE=TX_SAMPLING_RATE,
                     PREAMBLE=PREAMBLE)

    down_signal = rx.downconvert_signal(rf_signal)

    dec_signal = rx.resample(down_signal)

    packet_start_ind = rx.preamble_detection(dec_signal)
    dec_signal = dec_signal[packet_start_ind:]
    preamble_start_ind, packet_end_index = rx.preamble_synchronization(packet_start_ind, dec_signal)
    dec_signal = dec_signal[preamble_start_ind:]

    downChirp_series = rx.chirp_maker(len(dec_signal))

    # dechirp of the packet
    dechirp = dec_signal * downChirp_series

    preamble_header_symbols, payload_length = rx.fft_decode_preample_header(dechirp, mode=1)

    rx_out_symbols = rx.fft_decode_payload(preamble_header_symbols, dechirp, payload_length)
    if mode == 1:
        gray_rx_symbols = np.zeros(len(rx_out_symbols), dtype=int)
        for i in range(len(rx_out_symbols)):
            gray_rx_symbols[i] = rx.gray_encoder(int(rx_out_symbols[i]))
        print("Gray DeCoding...symbols", gray_rx_symbols)

    if mode == 1:
        rx_symbols = gray_rx_symbols
    else:
        rx_symbols = rx_out_symbols

    if np.array_equal(rx_symbols, tx_symbols):
        print("!!!!!!!!!!!!!!Correct Packet!!!!!!!!!!!!!!!!!")
    else:
        print("***************Wrong Packet******************")

    # while True:
    #    for i in range(10):
    #        keyboard.press(Key.media_volume_up)
    #        keyboard.release(Key.media_volume_up)
    #        time.sleep(0.1)
    #    for i in range(10):
    #        keyboard.press(Key.media_volume_down)
    #       keyboard.release(Key.media_volume_down)
    #        time.sleep(0.1)
    #    time.sleep(2)
    # dec_signal = rx.return_signal()

    # fig = plt.figure(1)
    # NFFT = 1024
    # #plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
    # plt.specgram(packet[:,1], NFFT=NFFT, Fs=192000, noverlap=900)
    # plt.title('Spectrogram')
    # plt.ylabel('Frequency band')
    # plt.xlabel('Time window')
    # plt.grid(True)
    # plt.draw()
    # plt.show(block=True)
    # plt.savefig("RX_Packet.pdf")

    # fig = plt.figure(2)
    # NFFT = 1024
    # Fs_new=5000
    # # #plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
    # plt.specgram(dec_signal, NFFT=NFFT, Fs=Fs_new, noverlap=900)
    # plt.title('Spectrogram')
    # plt.ylabel('Frequency band')
    # plt.xlabel('Time window')
    # plt.grid(True)
    # plt.draw()
    # plt.show(block=True)
