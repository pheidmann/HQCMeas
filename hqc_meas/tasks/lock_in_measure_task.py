# -*- coding: utf-8 -*-
"""
"""
from atom.api import (Enum, Float, observe, set_default)

from time import sleep

from .instr_task import InstrumentTask
from .tools.task_decorator import smooth_instr_crash

class LockInMeasureTask(InstrumentTask):
    """Ask a lock-in to perform a measure. Wait for any parallel operation
    before execution.
    """
    mode = Enum('X', 'Y', 'X&Y', 'Amp', 'Phase',
                         'Amp&Phase').tag(pref = True)
    waiting_time = Float().tag(pref = True)

    driver_list = ['SR7265-LI', 'SR7270-LI', 'SR830']

    task_database_entries = set_default({'x' : 1.0})

    def __init__(self, **kwargs):
        super(LockInMeasureTask, self).__init__(**kwargs)
        self.make_wait(wait = ['instr'])
        
    @smooth_instr_crash
    def process(self):
        """
        """
        if not self.driver:
            self.start_driver()

        sleep(self.waiting_time)

        if self.mode == 'X':
            value = self.driver.read_x()
            self.write_in_database('x', value)
        elif self.mode == 'Y':
            value = self.driver.read_y()
            self.write_in_database('y', value)
        elif self.mode == 'X&Y':
            value_x, value_y = self.driver.read_xy()
            self.write_in_database('x', value_x)
            self.write_in_database('y', value_y)
        elif self.mode == 'Amp':
            value = self.driver.read_amplitude()
            self.write_in_database('amplitude', value)
        elif self.mode == 'Phase':
            value = self.driver.read_phase()
            self.write_in_database('phase', value)
        elif self.mode == 'Amp&Phase':
            amplitude, phase = self.driver.read_amp_and_phase()
            self.write_in_database('amplitude', amplitude)
            self.write_in_database('phase', phase)

    @observe('mode')
    def _update_database_entries(self, change):
        """
        """
        new = change['value']
        if new == 'X':
            self.task_database_entries = {'x' : 1.0}
        elif new == 'Y':
            self.task_database_entries = {'y' : 1.0}
        elif new == 'X&Y':
            self.task_database_entries = {'x' : 1.0, 'y' : 1.0}
        elif new == 'Amp':
            self.task_database_entries = {'amplitude' : 1.0}
        elif new == 'Phase':
            self.task_database_entries = {'phase' : 1.0}
        elif new == 'Amp&Phase':
            self.task_database_entries = {'amplitude' : 1.0, 'phase' : 1.0}