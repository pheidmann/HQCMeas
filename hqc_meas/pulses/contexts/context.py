# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 15:14:02 2014

@author: Pierre
"""
from atom.api import Enum, Float
import numpy as np
from inspect import cleandoc
from hqc_meas.utils.atom_util import HasPrefAtom


class Context(HasPrefAtom):

    channels = Enum('Ch1_A', 'Ch1_M1', 'Ch1_M2',
                    'Ch2_A', 'Ch2_M1', 'Ch2_M2',
                    'Ch3_A', 'Ch3_M1', 'Ch3_M2',
                    'Ch4_A', 'Ch4_M1', 'Ch4_M2')

    time_unit = Enum('mus', 's', 'ms', 'ns')
    samplingfreq = Float()

    def compile_sequence(pulses, **kwargs):
        """ From a sequence of pulses, it creates a bytearray to communicate
        with AWG5014B

        """
        # Total length of the sequence to send to the AWG
        if 'sequence_duration' in kwargs:
            sequence_duration = kwargs['sequence_duration']
        else:
            sequence_duration = max([pulse.stop for pulse in pulses])

        # Collect the channels used in the pulses' sequence
        used_channels = set([pulse.channel[:3] for pulse in pulses])

        # Coefficient to convert the start and stop of pulses in second and
        # then in index integer for array
        if self.time_unit == 's':
            timetosec_coef = 1
        elif self.time_unit == 'ms':
            timetosec_coef = 1E-3
        elif self.time_unit == 'mus':
            timetosec_coef = 1E-6
        elif self.time_unit == 'ns':
            timetosec_coef = 1E-9
        timetoindex_coef = timetosec_coef * self.samplingfreq
        # Length of the sequence
        sequence_length = sequence_duration * timetoindex_coef
        # create 3 array for each used_channels
        array_analog = {}
        array_M1 = {}
        array_M2 = {}
        for channel in used_channels:
            # numpy array for analog channels int16 init 2**13
            array_analog[channel] = np.ones(sequence_length,
                                            dtype=np.uint16)*(2**13)
            # numpy array for marker1 int8 init 0. For AWG M1 = 0 = off
            array_M1[channel] = np.zeros(sequence_length, dtype=np.int8)
            # numpy array for marker2 int8 init 0. For AWG M2 = 0 = off
            array_M2[channel] = np.ones(sequence_length, dtype=np.int8)

        for pulse in pulses:
            startindex = int(pulse.start*timetoindex_coef)
            stopindex = int(pulse.stop*timetoindex_coef) + 1  # +1 considering
            # that the last value to write is at the position of stopindex
            pulsesize = stopindex - startindex
            channel = pulse.channel[:3]
            channeltype = pulse.channel[4:]

            if channeltype == 'A':
                array_analog[channel][startindex:stopindex] +=\
                    np.rint(8191*pulse.waveform)
            elif channeltype == 'M1':
                array_M1[channel][startindex:stopindex] +=\
                    np.ones(pulsesize, dtype=np.int8)
            elif channeltype == 'M2':
                array_M2[channel][startindex:stopindex] -=\
                    np.ones(pulsesize, dtype=np.int8)

        # Check the overflows
        for channel in used_channels:
            if array_analog[channel].max() > 16383 or\
                    array_analog[channel].min() < 0:
                return cleandoc('''analogical values out of allowed values
                                in channel {}'''.format(channel))
            elif array_M1[channel].max() > 1 or array_M1[channel].min() < 0:
                return cleandoc('''Overflow in Marker 1 of channel {}'''
                                .format(channel))
            elif array_M2[channel].max() > 1 or array_M2[channel].min() < 0:
                return cleandoc('''Overflow in Marker 2 of channel {}'''
                                .format(channel))

        # total numpy array to send to the AWG
        array = {}
        for channel in used_channels:
            array[channel] = array_analog[channel] +\
                array_M1[channel]*(2**14) + array_M2[channel]*(2**15)

        # byte array to send to the AWG
        tosend_array = {}
        for channel in used_channels:
            tosend_array[channel] = np.empty(2*sequence_length,
                                             dtype=np.int8)
            tosend_array[channel][::2] = array[channel] % 2**8
            tosend_array[channel][1::2] = array[channel] // 2**8

        return tosend_array
