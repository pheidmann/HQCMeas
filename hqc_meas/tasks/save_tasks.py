# -*- coding: utf-8 -*-
"""
"""

from atom.api import (Instance, ContainerList, Str, Enum, Value, Atom,
                      Bool, Int, observe, set_default, Unicode, Callable)

import os, numpy
from inspect import cleandoc

from .tools.database_string_formatter import (get_formatted_string,
                                              format_and_eval_string)
from .base_tasks import SimpleTask

class SavedValue(Atom):
    """
    """
    label = Str()
    value = Str()
    updater = Callable()

class SaveTask(SimpleTask):
    """Save the specified entries either in a CSV file or an array. Wait for any
    parallel operation before execution.
    """
    folder = Unicode().tag(pref = True)
    filename = Str().tag(pref = True)
    file_object = Value
    header = Str().tag(pref = True)

    array = Value() #Array

    saving_target = Enum('File', 'Array', 'File and array').tag(pref = True)

    array_size = Str().tag(pref = True)
    array_length = Int()
    line_index = Int(0)

    saved_values = ContainerList(Instance(SavedValue))

    initialized = Bool(False)
    task_database_entries = set_default({'file' : None})

    def __init__(self, **kwargs):
        super(SaveTask, self).__init__(**kwargs)
        self.make_wait()
        
    def process(self):
        """
        """
        #Init
        if not self.initialized:
            self.array_length = format_and_eval_string(self.array_size,
                                                       self.task_path,
                                                       self.task_database)
            if self.saving_target != 'Array':
                full_folder_path = get_formatted_string(self.folder,
                                                        self.task_path,
                                                        self.task_database)
                                                        
                filename = get_formatted_string(self.filename,
                                                self.task_path,
                                                self.task_database)
                                                
                full_path = os.path.join(full_folder_path, filename)
                try:
                    self.file_object = open(full_path, 'w')
                except IOError:
                    print 'In {}, failed to open the specified file'.format(
                                                                self.task_name)
                    self.root_task.should_stop.set()
                    return

                self.write_in_database('file', self.file_object)
                if self.header:
                    for line in self.header.split('\n'):
                        self.file_object.write('# ' + line + '\n')
                labels = [s.label for s in self.saved_values]
                self.file_object.write('\t'.join(labels) + '\n')
                self.file_object.flush()

            if self.saving_target != 'File':
                # TODO add more flexibilty on the dtype (possible complex values)
                array_type = numpy.dtype([(str(s.label), 'f8')
                                            for s in self.saved_values])
                self.array = numpy.empty((self.array_length),
                                         dtype = array_type)
                self.write_in_database('array', self.array)
            self.initialized = True

        #writing
        values = [format_and_eval_string(s.value,
                                       self.task_path,
                                       self.task_database)
                    for s in self.saved_values]
        if self.saving_target != 'Array':
            self.file_object.write('\t'.join([str(val)
                                              for val in values]) + '\n')
            self.file_object.flush()
        if self.saving_target != 'File':
            self.array[self.line_index] = tuple(values)

        self.line_index += 1

        #Closing
        if self.line_index == self.array_length:
            self.write_in_database('array', self.array)
            self.file_object.close()
            self.initialized = False

    def check(self, *args, **kwargs):
        """
        """
        traceback = {}
        try:
            full_folder_path = get_formatted_string(self.folder,
                                                         self.task_path,
                                                         self.task_database)
        except:
            traceback[self.task_path + '/' +self.task_name] = \
                'Failed to format the folder path'
            return False, traceback

        try:
            filename = get_formatted_string(self.filename, self.task_path,
                                                         self.task_database)
        except:
            traceback[self.task_path + '/' +self.task_name] = \
                'Failed to format the filename'
            return False, traceback

        full_path = os.path.join(full_folder_path, filename)

        try:
            f = open(full_path, 'wb')
            f.close()
        except:
            traceback[self.task_path + '/' +self.task_name] = \
                'Failed to open the specified file'
            return False, traceback

        try:
            format_and_eval_string(self.array_size,
                                       self.task_path,
                                       self.task_database)
        except:
            traceback[self.task_path + '/' +self.task_name] = \
                'Failed to compute the array size'
            return False, traceback

        test = True
        for i, s in enumerate(self.saved_values):
            try:
                format_and_eval_string(s.value,
                                   self.task_path,
                                   self.task_database)
            except:
                traceback[self.task_path + '/' + self.task_name + str(i)] = \
                    'Failed to evaluate entry : {}'.format(s.label)
                test = False

        if self.saving_target != 'File':
            data = [numpy.array([0.0,1.0]) for s in self.saved_values]
            names = str(','.join([s.label for s in self.saved_values]))
            final_arr = numpy.rec.fromarrays(data, names = names)

            self.write_in_database('array', final_arr)

        return test, traceback

    def update_preferences_from_members(self):
        """
        """
        super(SaveTask, self).update_preferences_from_members()
        self.task_preferences['saved_values'] = [(s.label, s.value) 
                                                for s in self.saved_values]

    def update_members_from_preferences(self, **parameters):
        """
        """
        super(SaveTask, self).update_members_from_preferences(**parameters)
        if 'saved_values' in parameters:
            self.saved_values = [SavedValue(label = s[0], value = s[1])
                                    for s in parameters['saved_values']]

    @observe('saving_target')
    def _update_database_entries(self, new):
        """
        """
        if new == 'File':
            self.task_database_entries = {'file' : None}
        elif new == 'Array':
            self.task_database_entries = {'array' : numpy.array([1.0])}
        else:
            self.task_database_entries = {'file' : None,
                                          'array' : numpy.array([1.0])}

class SaveArrayTask(SimpleTask):
    """Save the specified array either in a CSV file or as a .npy binary file.
    Wait for any parallel operation before execution.
    """
    folder = Unicode().tag(pref = True)
    filename = Str().tag(pref = True)
    file_object = Value()
    header = Str().tag(pref = True)

    target_array = Str().tag(pref = True)
    mode = Enum('Text file', 'Binary file').tag(pref = True)

    def __init__(self, **kwargs):
        super(SaveArrayTask, self).__init__(**kwargs)
        self.make_wait()

    def process(self):
        """
        """
        array_to_save = self.get_from_database(self.target_array[1:-1])

        full_folder_path = get_formatted_string(self.folder,
                                                self.task_path,
                                                self.task_database)
                                                
        filename = get_formatted_string(self.filename,
                                        self.task_path,
                                        self.task_database)
                                        
        full_path = os.path.join(full_folder_path, filename)

        if self.mode == 'Text file':
            try:
                self.file_object = open(full_path, 'wb')
            except IOError:
                print 'In {}, failed to open the specified file'.format(
                                                            self.task_name)
                self.root_task.should_stop.set()
                return

            if self.header:
                for line in self.header.split('\n'):
                    self.file_object.write('# ' + line + '\n')
            if array_to_save.dtype.names:
                self.file_object.write('\t'.join(array_to_save.dtype.names) + \
                                        '\n')
            numpy.savetxt(self.file_object, array_to_save, delimiter = '\t')
            self.file_object.close()

        else:
            try:
                self.file_object = open(full_path, 'wb')
                self.file_object.close()
            except IOError:
                print 'In {}, failed to open the specified file'.format(
                                                            self.task_name)
                self.root_task.should_stop.set()
                return

            numpy.save(full_path, array_to_save)

    def check(self, *args, **kwargs):
        """
        """
        traceback = {}
        try:
            full_folder_path = get_formatted_string(self.folder,
                                                         self.task_path,
                                                         self.task_database)
        except:
            traceback[self.task_path + '/' +self.task_name] = \
                'Failed to format the folder path'
            return False, traceback

        if self.mode == 'Binary file':
            if len(self.filename) > 3:
                if self.filename[-4] == '.' and self.filename[-3:] != 'npy':
                    self.filename = self.filename[:-4] + '.npy'
                    print cleandoc("""The extension of the file will be replaced
                        by '.npy' in task {}""".format(self.task_name))

        try:
            filename = get_formatted_string(self.filename, self.task_path,
                                                         self.task_database)
        except:
            traceback[self.task_path + '/' +self.task_name] = \
                'Failed to format the filename'
            return False, traceback

        full_path = os.path.join(full_folder_path, filename)

        try:
            f = open(full_path, 'wb')
            f.close()
        except:
            traceback[self.task_path + '/' +self.task_name] = \
                'Failed to open the specified file'
            return False, traceback

        entries = self.task_database.list_accessible_entries(self.task_path)
        if self.target_array[1:-1] not in entries:
            traceback[self.task_path + '/' +self.task_name] = \
                'Specified array is absent from the database'
            return False, traceback

        return True, traceback