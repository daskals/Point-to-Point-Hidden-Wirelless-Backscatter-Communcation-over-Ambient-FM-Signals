#######################################################
#     Spiros Daskalakis                               #
#     last Revision: 19/03/2021                       #
#     Python Version:  3.9                            #
#     Email: daskalakispiros@gmail.com                #
#     Website: www.daskalakispiros.com                #
#######################################################
import datetime

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from messenger import Ui_MainWindow

import yaml
from box import Box
from rtlsdr import RtlSdr
import sys

from TX import TXtag
from RX import RX_receiver

from matplotlib import pyplot as plt
import numpy as np


class Tag_transmitter(QObject):
    # This defines a signal called 'finished' that takes no arguments.
    finished = pyqtSignal()
    data_TX_out = pyqtSignal(np.ndarray)

    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

    def __init__(self, mode=cfg.MODE, CR=cfg.CR, SF=cfg.SF, CHIRP_F_START=cfg.CHIRP_F_START,
                 CHIRP_F_STOP=cfg.CHIRP_F_STOP):
        super(Tag_transmitter, self).__init__()

        self.working = True
        self.text = 'LORAB'
        self.BW = CHIRP_F_STOP - CHIRP_F_START

        with open("config.yml", "r") as ymlfile:
            cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

        self.TX_SAMPLING_RATE = cfg.TX_SAMPLING_RATE
        self.SOUND_RES = cfg.SOUND_RES
        self.PREAMBLE = cfg.PREAMBLE

        self.tag = TXtag(mode=mode, CR=CR, SF=SF, BW=self.BW, CHIRP_F_START=CHIRP_F_START,
                         TX_SAMPLING_RATE=self.TX_SAMPLING_RATE,
                         SOUND_RES=self.SOUND_RES, PREAMBLE=self.PREAMBLE)

    @pyqtSlot()
    def work(self):
        """Long-running task."""
        i = 0
        after_text = 0
        while self.working:
            # send the same packet many times
            if self.text != after_text:
                # print('******************************TAG Running Loop**************************')
                # if the text message is different then create a new packet
                header_symbols, symbols, tx_symbols = self.tag.create_symbols(self.text)
                packet = self.tag.create_packet(header_symbols, symbols)
                after_text = self.text
            # send packet to sound card
            rf_signal_tx = packet[:, 0] + 1j * packet[:, 1]

            self.tag.send_packet_to_sound_card(packet)
            # self.data_TX_out.emit(rf_signal_tx)
            # print(packet[:,0])

        self.finished.emit()


class SDR_receiver(QObject):
    # This defines a signal called 'finished' that takes no arguments.
    finished = pyqtSignal()
    data_RX_out = pyqtSignal(np.ndarray)

    with open("config.yml", "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

    def __init__(self, CR=cfg.CR, SF=cfg.SF, CHIRP_F_START=cfg.CHIRP_F_START, CHIRP_F_STOP=cfg.CHIRP_F_STOP,
                 mode=cfg.MODE, SDR_FREQ=95800000, SDR_GAIN=10, SFO_CHECK=False,
                 AGC_CKECK=False):
        super(SDR_receiver, self).__init__()

        with open("config.yml", "r") as ymlfile:
            cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

        self.sample_rate = cfg.RX_SAMPLING_RATE

        self.sdr = RtlSdr()
        self.sdr.sample_rate = self.sample_rate
        self.sdr.freq_correction = 60  # PPM
        self.sdr.center_freq = SDR_FREQ
        self.sdr.gain = SDR_GAIN  # 'auto'

        self.BW = CHIRP_F_STOP - CHIRP_F_START
        self.working = True

        self.PREAMBLE = cfg.PREAMBLE
        self.mode = mode

        self.t_sampling = 0.5  # Sampling 0.5 sec
        self.N_samples = round(self.sample_rate * self.t_sampling)

        self.rx = RX_receiver(mode=mode, CR=CR, SF=SF, BW=self.BW, CHIRP_F_START=CHIRP_F_START, PREAMBLE=self.PREAMBLE,
                              RX_SAMPLING_RATE=self.sample_rate)

    def work(self):
        """Long-running task."""
        while self.working:
            samples = self.sdr.read_samples(self.N_samples)
            # Convert samples to a numpy array
            rf_signal = np.array(samples).astype("complex64")
            down_signal = self.rx.downconvert_signal(rf_signal)
            dec_signal = self.rx.resample(down_signal)

            print('RX Original Signal Length: ', len(rf_signal))
            print('RX Resample Signal Length: ', len(dec_signal))

            packet_start_ind = self.rx.preamble_detection(dec_signal)

            dec_signal = dec_signal[packet_start_ind:]
            preamble_start_ind, header_EndIndex = self.rx.preamble_synchronization(packet_start_ind, dec_signal)

            if packet_start_ind < 0 or preamble_start_ind < 0:
                print('WARNING!! Can not find the FRAME start')
                continue
            if packet_start_ind > len(dec_signal) or preamble_start_ind > len(dec_signal):
                print('WARNING!! Packet cut in the middle')
                continue
            dec_signal = dec_signal[preamble_start_ind:]

            downChirp_series = self.rx.chirp_maker(len(dec_signal))

            # dechirp of the packet
            dechirp = dec_signal * downChirp_series

            # preamble_header_symbols, payload_length = self.rx.fft_decode_preample_header(dechirp, mode=self.mode)

            # rx_out_symbols = self.rx.fft_decode_payload(preamble_header_symbols, dechirp, payload_length)

            print('******************************SDR Receiver Running Loop**************************')
            self.data_RX_out.emit(rf_signal)
        self.sdr.close()
        self.finished.emit()


class MessengerWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, connected=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load configuration file config.yml
        with open("config.yml", "r") as ymlfile:
            self.cfg = Box(yaml.safe_load(ymlfile), default_box=True, default_box_attr=None)

        self.setupUi(self)
        self.max_fm_station_freq = 0
        self.figure1 = plt.figure(1)
        self.figure2 = plt.figure(2)

        # send message
        self.pushButton.pressed.connect(self.button_send)
        # connect with RX and TX systems
        self.pushButton_2.pressed.connect(self.connect)

        self.pushButton_Find_FM.pressed.connect(self.find_FM)

    def find_FM(self):
        """
                        This function scans the FM band 87.5 to 109 Mhz in order to find the FM station with the maximum
                        power
                        the function ends with setting the SDR frequency with the above finding value
        """
        sdr = RtlSdr()
        # configure device
        Fs = 2e6  # Hz
        sdr.sample_rate = Fs  # Hz
        sdr.freq_correction = 60  # PPM
        sdr.gain = 'auto'

        FM_band_min = 87.5
        FM_band_max = 109
        powe = np.ndarray(0)
        freq = np.ndarray(0)

        t_sampling = 0.1  # Sampling for 100 ms
        N_samples = round(Fs * t_sampling)

        for i in np.arange(FM_band_min, FM_band_max, Fs / 1e6):
            sdr.center_freq = i * 1e6  # Hz
            counter = 0
            prev_int = 0
            while 1:
                counter = counter + 1
                samples = sdr.read_samples(N_samples)
                ###################################################################################
                power, psd_freq = plt.psd(samples, NFFT=1024, Fs=sdr.sample_rate / 1e6, Fc=sdr.center_freq / 1e6)
                #####################################################################################
                ind_pow = np.argmax(power)
                freq_ind = round(psd_freq[ind_pow], 1)
                if freq_ind == prev_int and counter >= 3:
                    powe = np.append(powe, ind_pow)
                    freq = np.append(freq, freq_ind)
                    break
                prev_int = freq_ind
        # done
        sdr.close()
        max_fm_station_power = np.argmax(powe)
        max_fm_station_freq = freq[max_fm_station_power]
        print('FM with max Power: ' + str(max_fm_station_freq) + ' MHz')
        # self.pushButton_Find_FM.setText('FM with max Power: '+str(max_fm_station_freq)+' MHz')
        self.spinBox_SDR_Freq.setValue(int(max_fm_station_freq * 1e6))

    def connect(self):
        # get a list of all speakers:
        # print(sc.all_speakers())
        # if len(sc.all_speakers()) == 1:
        #    reply = QMessageBox.warning(self, 'TX Warning', 'Connect the RX to audio Jack !!!')
        #    return

        if self.pushButton_2.text() == 'Connect':
            # rename button to Disconnect
            self.pushButton_2.setText('Disconnect')
            # Deactivate all the settings
            self.deactivate_settings()
            # take values from spinboxes

            # take values from spinboxes
            CR = int(self.spinBox_CR.value())
            MODE = int(self.spinBox_Mode.value())
            SF = int(self.spinBox_SF.value())
            CHIRP_F_START = int(self.spinBox_Fmin.value())
            CHIRP_F_STOP = int(self.spinBox_Fmax.value())
            SDR_FREQ = int(self.spinBox_SDR_Freq.value())
            SDR_GAIN = int(self.spinBox_SDR_gain.value())
            SFO_CHECK = int(self.CFO_check.checkState())
            AGC_CKECK = int(self.AGC_check.checkState())

            ####################################################################
            # Step 2: Create a QThread object
            self.thread = QThread()
            # Step 3: Create a worker object
            self.worker = SDR_receiver(CR=CR, SF=SF, CHIRP_F_START=CHIRP_F_START, CHIRP_F_STOP=CHIRP_F_STOP,
                                       mode=MODE, SDR_FREQ=SDR_FREQ, SDR_GAIN=SDR_GAIN, SFO_CHECK=SFO_CHECK,
                                       AGC_CKECK=AGC_CKECK)
            # Step 4: Move worker to the thread
            self.worker.moveToThread(self.thread)
            # Step 5: Connect signals and slots
            # begin our worker object's loop when the thread starts running
            self.thread.started.connect(self.worker.work)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.worker.data_RX_out.connect(self.print_RX_plot)
            # self.worker.progress.connect(self.reportProgress)
            # Step 6: Start the thread
            self.thread.start()

            #####################################################################
            # Step 2: Create a QThread object
            self.thread2 = QThread()
            # Step 3: Create a worker object
            self.worker2 = Tag_transmitter(CR=CR, SF=SF, CHIRP_F_START=CHIRP_F_START, CHIRP_F_STOP=CHIRP_F_STOP,
                                           mode=MODE)
            # Step 4: Move worker to the thread
            self.worker2.moveToThread(self.thread2)
            # Step 5: Connect signals and slots
            # begin our worker object's loop when the thread starts running
            self.thread2.started.connect(self.worker2.work)
            self.worker2.finished.connect(self.thread2.quit)
            self.worker2.finished.connect(self.worker2.deleteLater)
            self.thread2.finished.connect(self.thread2.deleteLater)

            # The thread returns back two signals with the current measurement
            # Update the current monitors of the GUI
            self.worker2.data_TX_out.connect(self.print_TX_plot)

            # Step 6: Start the thread
            self.thread2.start()

        elif self.pushButton_2.text() == 'Disconnect':
            self.pushButton_2.setText('Connect')
            # Stop the thread if pressed the Disconnect Button
            self.worker.working = False
            self.worker2.working = False

            self.activate_settings()

        # if self.pushButton_2.pressed:
        # input_text = query.lower()
        # self.textEdit.append(input_text)

    def print_RX_plot(self, *value):
        # in mA
        # print(value[0])
        NFFT = 1024
        #plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
        plt.specgram(value[0], NFFT=NFFT, Fs=1140000, noverlap=900)
        plt.title('Spectrogram')
        plt.ylabel('Frequency band')
        plt.xlabel('Time window')
        plt.draw()
        plt.pause(0.1)
        self.figure2.clear()

    def print_TX_plot(self, *value):
        # in mA
        print(value[0])
        NFFT = 1024
        #plt.imshow(spectrogram, aspect='auto', cmap='hot_r', origin='lower')
        plt.specgram(value[0], NFFT=NFFT, Fs=192000, noverlap=900)
        plt.title('Spectrogram')
        plt.ylabel('Frequency band')
        plt.xlabel('Time window')
        plt.draw()
        plt.pause(0.1)
        self.figure1.clear()



    def show_text(self, text):
        self.textBrowser.append(text)
        self.textBrowser.repaint()

    def print_message(self, message):
        username = message['username']
        message_time = message['time']
        text = message['text']
        dt = datetime.fromtimestamp(message_time)
        dt_beauty = dt.strftime('%H:%M:%S')

        self.show_text(f'{dt_beauty} {username}\n{text}\n\n')

    def deactivate_settings(self):
        # Deactivate all the settings
        self.spinBox_SDR_gain.setEnabled(False)
        self.spinBox_SDR_Freq.setEnabled(False)
        self.spinBox_CR.setEnabled(False)
        self.spinBox_Fmax.setEnabled(False)
        self.spinBox_Fmin.setEnabled(False)
        self.spinBox_SF.setEnabled(False)
        self.spinBox_Mode.setEnabled(False)
        self.pushButton_Find_FM.setEnabled(False)
        self.CFO_check.setEnabled(False)
        self.AGC_check.setEnabled(False)
        # labels
        self.fmax_label.setEnabled(False)
        self.fmin_label.setEnabled(False)
        self.mode_label.setEnabled(False)
        self.sf_label.setEnabled(False)
        self.cr_label.setEnabled(False)
        self.freq_label.setEnabled(False)
        self.gain_label.setEnabled(False)

    def activate_settings(self):
        # Activate all Settings
        self.spinBox_SDR_gain.setEnabled(True)
        self.spinBox_SDR_Freq.setEnabled(True)
        self.spinBox_CR.setEnabled(True)
        self.pushButton_Find_FM.setEnabled(True)
        self.spinBox_Fmax.setEnabled(True)
        self.spinBox_Fmin.setEnabled(True)
        self.spinBox_SF.setEnabled(True)
        self.spinBox_Mode.setEnabled(True)
        self.CFO_check.setEnabled(True)
        self.AGC_check.setEnabled(True)
        # labels
        self.fmax_label.setEnabled(True)
        self.fmin_label.setEnabled(True)
        self.mode_label.setEnabled(True)
        self.sf_label.setEnabled(True)
        self.cr_label.setEnabled(True)
        self.freq_label.setEnabled(True)
        self.gain_label.setEnabled(True)

    def closeEvent(self, event):
        if self.pushButton_2.text() == 'Disconnect':
            reply = QMessageBox.warning(self, 'Window Close',
                                        'The system is still connected with the MiniBB. Disconnect First!!!')
            event.ignore()

    def button_send(self):
        # if its not connected then set popup a window
        if self.pushButton_2.text() == 'Connect':
            reply = QMessageBox.warning(self, 'Window Close',
                                        'Connect First!!!')
            return 0

        text = self.textEdit.toPlainText()
        self.worker2.text = text
        self.send_message(text)
        self.textEdit.setText('')
        self.textEdit.repaint()

    def send_message(self, text):
        self.show_text('Me: ' + text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MessengerWindow()
    window.show()
    sys.exit(app.exec_())
