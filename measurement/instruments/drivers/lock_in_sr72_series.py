# -*- coding: utf-8 -*-
"""

"""

from visa import Instrument, VisaIOError

class LockInSR72Series(Instrument):
    """
    """

    def __init__(self, *args, **kwargs):

        if not kwargs.has_key('term_chars'):
            kwargs['term_chars'] = '\0'
        super(LockInSR72Series, self).__init__(*args, **kwargs)

    def read_x(self):
        """
        """
        value = self.ask_for_values('X.')[0]
        status = self._check_status()
        if status != 'OK':
            raise VisaIOError('The command did not complete correctly')
        else:
            return value

    def read_y(self):
        """
        """
        value = self.ask_for_values('Y.')[0]
        status = self._check_status()
        if status != 'OK':
            raise VisaIOError('The command did not complete correctly')
        else:
            return value

    def read_xy(self):
        """
        """
        values = self.ask_for_values('XY.')
        status = self._check_status()
        if status != 'OK':
            raise VisaIOError('The command did not complete correctly')
        else:
            return values

    def read_amplitude(self):
        """
        """
        value = self.ask_for_values('MAG.')[0]
        status = self._check_status()
        if status != 'OK':
            return status
        else:
            return value

    def read_phase(self):
        """
        """
        value = self.ask_for_values('PHA.')[0]
        status = self._check_status()
        if status != 'OK':
            raise VisaIOError('The command did not complete correctly')
        else:
            return value

    def read_amp_and_phase(self):
        """
        """
        values = self.ask_for_values('MP.')
        status = self._check_status()
        if status != 'OK':
            raise VisaIOError('The command did not complete correctly')
        else:
            return values

    def _check_status(self):
        """
        """
        bites = self.read()
        status_byte = ('{0:08b}'.format(ord(bites[0])))[::-1]
        if not status_byte[0]:
            return 'Command went wrong'
        else:
            return 'OK'