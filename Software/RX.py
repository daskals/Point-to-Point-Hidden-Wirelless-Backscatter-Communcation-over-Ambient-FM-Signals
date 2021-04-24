#######################################################
#     Spiros Daskalakis                               #
#     last Revision: 16/03/2021                       #
#     Python Version:  3.9                            #
#     Email: daskalakispiros@gmail.com                #
#     Website: www.daskalakispiros.com                #
#######################################################

import numpy as np
import scipy.signal as sps
import coding
from matplotlib import pyplot as plt


class RX_receiver:

    def __init__(self, mode=0, num_symbols=0, CR=4, SF=8, BW=5000, CHIRP_F_START=15000, RX_SAMPLING_RATE=192000,
                 PREAMBLE=[-1, -1, -1, 1, 1, 1, 1]):

        self.CR = CR
        self.SF = SF
        self.BW = BW
        self.CHIRP_F_START = CHIRP_F_START
        self.RX_SAMPLING_RATE = RX_SAMPLING_RATE
        self.print_rx_info()

        preamble_mask = np.array(PREAMBLE)

        self.index_count_1 = 0
        self.index_start = 0
        self.index_end = len(preamble_mask) - 1

        for i in range(len(preamble_mask)):
            if preamble_mask[i] == 1:
                self.index_count_1 = self.index_count_1 + 1
                if self.index_count_1 == 1:
                    self.index_start = i

        f0 = -self.BW / 2
        self.preamble_first_chirp = self.my_chirp(self.BW, 1, f0)
        # first_chirp =self.lora_symbol(shift=0, inverse=1)
        # self.preamble_first_chirp = first_chirp[:, 0] + 1j * first_chirp[:, 1]

        self.preamble_symbol_chirp = self.my_chirp(self.BW, 0, f0)
        self.preamble_symbol_chirp_series = np.tile(self.preamble_symbol_chirp, self.index_count_1)
        # first_chirp2 =self.lora_symbol(shift=0, inverse=0)
        # self.preamble_symbol_chirp = first_chirp2[:, 0] + 1j * first_chirp2[:, 1]

        self.symbols_per_preamble_header = self.index_count_1 + 2

        # self.symbols_per_frame = len(preamble_mask) + num_symbols
        # print(self.symbols_per_frame)
        self.T_symbol = (2 ** self.SF) / self.BW
        # self.total_packet_duration = self.symbols_per_frame * self.T_symbol

        ##################################################################
        self.new_Fs = self.BW
        self.over = int(self.T_symbol * self.new_Fs)

    def preamble_detection(self, rf_signal):
        # correlation for preamble finding
        corr = np.correlate(rf_signal, self.preamble_first_chirp, "full")
        # corr = corr[0: -1 - len(self.preamble_first_chirp)]
        corr = abs(corr)  # -np.mean(abs(corr))
        signalStartIndex = np.argmax(corr) + 1
        print('Index Packet start:', signalStartIndex)

        return signalStartIndex

    def preamble_synchronization(self, packetstart, rf_signal):

        # rf_signal = rf_signal[packetstart:]
        corr2 = np.correlate(rf_signal, self.preamble_symbol_chirp_series, "full")
        # corr2 = corr2[0: -1 - len(self.preamble_symbol_chirp_series)]
        corr2 = abs(corr2)  # -np.mean(abs(corr2))
        signalStartIndex = np.argmax(corr2) - self.index_count_1 * self.over + 1
        header_EndIndex = round(signalStartIndex + (self.index_count_1 + 2) * self.over + 1)
        print('Index Preamble start:', signalStartIndex)

        # plt.subplot(2, 1, 1)
        # plt.plot(rf_signal)
        # plt.subplot(2, 1, 2)
        # plt.plot(corr2)
        # plt.show()

        return signalStartIndex, header_EndIndex

    def fft_decode_preample_header(self, dechirp, mode=1):
        Fs = self.new_Fs
        over = int(self.T_symbol * Fs)
        symbols = np.zeros(self.symbols_per_preamble_header)  # create
        for m in range(self.symbols_per_preamble_header):
            signal = dechirp[(m * over): (m + 1) * over]
            FFT_out = np.abs(np.fft.fft(signal))
            r = np.max(FFT_out)
            c = np.argmax(FFT_out)
            print("Symbol#:\%d, Symbol:%d, Power:%d", m, c, r)
            symbols[m] = c

        preamble_header_symbols = symbols - round(np.mean(symbols[0:self.index_count_1 - 1])) % (
                2 ** self.SF)
        print(preamble_header_symbols)
        if mode == 1:
            payload_length = self.gray_encoder(int(preamble_header_symbols[-2]))
            CR = self.gray_encoder(int(preamble_header_symbols[-1]))
        else:
            payload_length = int(preamble_header_symbols[-2])
            CR = int(preamble_header_symbols[-1])

        print("###################END Preamble header")
        return symbols, payload_length

    def fft_decode_payload(self, symbols_pr_header, dechirp, payload_length):
        Fs = self.new_Fs
        over = int(self.T_symbol * Fs)
        symbols = np.zeros(payload_length)  # create
        for m in range(payload_length):
            signal = dechirp[((m + self.symbols_per_preamble_header) * over): ((
                                                                                       m + self.symbols_per_preamble_header) + 1) * over]
            FFT_out = np.abs(np.fft.fft(signal))
            r = np.max(FFT_out)
            c = np.argmax(FFT_out)
            print("Symbol#:\%d, Symbol:%d, Power:%d", m, c, r)
            symbols[m] = c
        symbols = symbols - round(np.mean(symbols_pr_header[0:self.index_count_1 - 1])) % (2 ** self.SF)
        return symbols

    def return_signal(self):
        return self.dec_signal

    def downconvert_signal(self, rf_signal):
        F_offset = self.CHIRP_F_START + self.BW / 2
        # To mix the data down, generate a digital complex exponential
        # (with the same length as x1) with phase -F_offset/Fs
        fc1 = np.exp(-1.0j * 2.0 * np.pi * F_offset / self.RX_SAMPLING_RATE * np.arange(len(rf_signal)))
        # Now, just multiply x1 and the digital complex expontential
        x_down = rf_signal * fc1
        return x_down

    def resample(self, downconv_signal):

        newrate = self.BW
        # channelize the signal
        ########################################################################
        samples = round(downconv_signal.size * newrate / self.RX_SAMPLING_RATE)
        resampled_signal1 = sps.resample(downconv_signal, samples)
        ########################################################################
        # dec_audio = int(self.RX_SAMPLING_RATE / newrate)
        # resampled_signal2 = sps.decimate(downconv_signal, dec_audio)
        ########################################################################
        # ratio = newrate / self.RX_SAMPLING_RATE
        # converter = 'sinc_best'  # or 'sinc_fastest', ...
        # resampled_signal3 = samplerate.resample(downconv_signal, ratio, converter)

        return resampled_signal1

    def chirp_maker(self, length):

        f0 = -self.BW / 2
        downChirp = self.my_chirp(self.new_Fs, 1, f0)
        # downChirp=self.lora_symbol(self.SF, self.BW, Fs, 0, 1, f0, 3)
        downChirp_series = np.tile(downChirp, int(np.ceil(length / len(downChirp))))
        downChirp_series = downChirp_series[:length]
        return downChirp_series

    def my_chirp(self, fs, inverse, left):
        sf = self.SF
        bw = self.BW

        symbol_time = 2 ** sf / bw
        k = bw / symbol_time

        t = np.arange(0, symbol_time, 1 / fs)
        if inverse == 0:
            if_chirp = 2 * np.pi * (left * t + k / 2 * t ** 2)
        elif inverse == 1:
            if_chirp = -2 * np.pi * (left * t + k / 2 * t ** 2)
        else:
            print("Error argument")
        chirp_out = np.cos(if_chirp) + 1j * np.sin(if_chirp)
        chirp_out = (1 + 1j) * chirp_out
        return chirp_out

    def print_rx_info(self):
        print("%%%%%%%%%%%% RX Parameters  %%%%%%%%%%%%%%%%")
        print("Sample Rate:", self.RX_SAMPLING_RATE, "Sps")
        print("Spreading factor:", self.SF)
        print("Coding rate:", self.CR)
        print("BandWith:", self.BW, "Hz")
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    def gray_encoder(self, n):

        """
        Convert given decimal number into decimal equivalent of its gray code form
        :param n: decimal number
        return: int
        """
        # Right Shift the number by 1 taking xor with original number
        return n ^ (n >> 1)

    def text_to_numbers(self):
        """
        Convert a string message to an integer number according the utf-8 encoding
        return: int numpy array
        """
        text = self.message_text
        # print("The original message is : " + str(text))
        arr = bytes(text, 'utf-8')
        numbers = np.zeros(len(arr), dtype=int)
        i = 0
        for byte in arr:
            numbers[i] = byte
            i = i + 1
        return numbers

    def lora_symbol(self, shift, inverse=0):
        # Initialization
        phase = 0
        f0 = -self.BW / 2
        Frequency_Offset = f0
        sf = self.SF
        fs = self.BW
        bw = self.BW
        num_samples_in = fs * (2 ** sf) / bw
        num_samples = round(num_samples_in)

        signal = np.zeros((num_samples, 2))

        for k in range(num_samples):
            # set output to cosine signal
            signal[k, 0] = np.cos(phase)
            signal[k, 1] = np.sin(phase)

            # ------------------------------------------
            # Frequency from cyclic shift
            f = bw * shift / (2 ** sf)
            if inverse == 1:
                f = bw - f
            # ------------------------------------------
            # apply Frequency offset away from DC
            f = f + Frequency_Offset
            # ------------------------------------------
            # Increase the phase according to frequency
            phase = phase + 2 * np.pi * f / fs
            if phase > np.pi:
                phase = phase - 2 * np.pi
            # ------------------------------------------
            # update cyclic shift
            shift = shift + bw / fs
            if shift >= (2 ** sf):
                shift = shift - 2 ** sf
        return signal

    def rx_encoder(self, RX_symbols, raw_symbols_tx):
        in_bit_matrix = coding.vec_bin_array(raw_symbols_tx, self.SF)
        in_bit_vector = in_bit_matrix.reshape(-1)
        N_bits_raw = len(in_bit_vector)
        # Round to a whole number of interleaving blocks
        N_bits = int(np.ceil(N_bits_raw / (4 * self.SF)) * (4 * self.SF))
        N_codewords = int(N_bits / 4)
        N_codedbits = N_codewords * (4 + self.CR)
        N_syms = int(N_codedbits / self.SF)

        gray_rx_symbols = np.zeros(len(RX_symbols), dtype=int)
        for i in range(len(RX_symbols)):
            gray_rx_symbols[i] = self.gray_encoder(int(RX_symbols[i]))
        print("Gray DeCoding...symbols", gray_rx_symbols)
        bit_matrix = coding.vec_bin_array(gray_rx_symbols, self.SF)
        # print("Matrix degray\n", bit_matrix)
        # ------------------------------------------------------
        Cest = coding.de_inter_liver(bit_matrix, N_codewords, N_codedbits, self.CR, self.SF)
        # print("DeInterliving Coding Cest matrix\n", Cest)

        bits_est = coding.Hamming_decode(Cest, N_bits, N_codewords, self.CR)
        ints = 0

        bit_matrix_final = np.zeros((raw_symbols_tx.size, self.SF), dtype=np.int32)
        for sym in range(0, bits_est.size, self.SF):
            bit_matrix_final[ints, :] = bits_est[sym: sym + self.SF]
            a = bit_matrix_final[ints, :]
            ints = ints + 1

        # print("bit final\n", bit_matrix_final)
        symbols_matrix_final = [coding.bool2int(x[::-1]) for x in bit_matrix_final]
        print("binary\n", symbols_matrix_final)
        if np.array_equal(raw_symbols_tx, symbols_matrix_final):
            print("!!!!!!!!!!!!!!Correct Packet!!!!!!!!!!!!!!!!!")
        else:
            print("***************Wrong Packet******************")
