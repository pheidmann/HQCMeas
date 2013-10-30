# -*- coding: utf-8 -*-
"""
"""

from traits.api\
    import (HasTraits, Str, Int, Instance, List, Bool, Type,
            on_trait_change, Unicode, Directory, BaseStr, BaseUnicode, Dict,
            Any)
from traits.api import self as trait_self
from traitsui.api\
     import (View, ListInstanceEditor, VGroup, HGroup, UItem,
             InstanceEditor)

from configobj import Section, ConfigObj
import os

from .tools.task_database import TaskDatabase
from .tools.task_decorator import make_stoppable

class AbstractTask(HasTraits):
    """Abstract  class defining common traits of all Task

    This class basically defines the minimal skeleton of a Task in term of
    traits and methods.
    """
    task_class = Str(preference = True)
    task_name = Str(preference = True)
    task_label = Str
    task_depth = Int
    task_preferences = Instance(Section)
    task_database = Instance(TaskDatabase)
    task_database_entries = Dict(Str, Any)
    task_path = Str

    #root_task = Instance(RootTask)

    def process(self):
        """The main method of any task as it is this one which is called when
        the measurement is performed. This method should always be decorated
        with make_stoppable.
        """
        err_str = 'This method should be implemented by subclasses of\
        AbstractTask. This method is called when the program requires the task\
        to be performed'
        raise NotImplementedError(err_str)

    def check(self, *args, **kwargs):
        """Method used to check that everything is alright before starting a
        measurement
        """
        err_str = 'This method should be implemented by subclasses of\
        AbstractTask. This method is called when the program requires the task\
        to check that all parameters are ok'
        raise NotImplementedError(err_str)

    def register_in_database(self):
        """Method used to create entries in the database
        """
        err_str = 'This method should be implemented by subclasses of\
        AbstractTask. This method is called when the program requires the task\
        to create its entry in the database'
        raise NotImplementedError(err_str)

    def unregister_from_database(self):
        """Method used to delete entries from the database
        """
        err_str = 'This method should be implemented by subclasses of\
        AbstractTask. This method is called when the program requires the task\
        to delete its entry from the database'
        raise NotImplementedError(err_str)

    def register_preferences(self):
        """Method used to create entries in the preferences object
        """
        err_str = 'This method should be implemented by subclasses of\
        AbstractTask. This method is called when the program requires the task\
        to create its entries in the preferences object'
        raise NotImplementedError(err_str)

    def update_preferences_from_traits(self):
        """Method used to update the entries in the preference object before
        saving
        """
        err_str = 'This method should be implemented by subclasses of\
        AbstractTask. This method is called when the program requires the task\
        to update the entries in the preference object before saving'
        raise NotImplementedError(err_str)

    def update_traits_from_preferences(self, **parameters):
        """Method used to update the trait values using the info extracted from
        a config file.

        Parameters:
        ----------
        parameters : dict
        """
        err_str = 'This method should be implemented by subclasses of\
        AbstractTask. This method is called when the program requires the task\
        to update update the trait values using the info extracted from\
        a config file.'
        raise NotImplementedError(err_str)

    def _task_class_default(self):
        """
        """
        return self.__class__.__name__

    def _task_name_changed(self, new):
        """
        """
        self.task_label = new + ' (' + self.task_class + ')'

    @on_trait_change('task_database_entries[]')
    def _update_database(self, obj, name, old, new):
        """
        """
        added = set(new) - set(old)
        removed = set(old) - set(new)
        if self.task_database:
            for entry in removed:
                self.remove_from_database(self.task_name + '_' + entry)
            for entry in added:
                self.write_in_database(entry, self.task_database_entries[entry])


class SimpleTask(AbstractTask):
    """Task with no child task, written in pure Python.
    """
    #Class attribute specifying if instances of that class can be used in loop
    # Not a Trait because otherwise would not be a class attribute
    loopable = False

    def write_in_database(self, name, value):
        """This method build a task specific database entry from the task_name
        and the name argument and set the database entry to the specified value.
        """
        value_name = self.task_name + '_' + name
        return self.task_database.set_value(self.task_path, value_name, value)

    def get_from_database(self, full_name):
        """This method return the value under the database entry specified by
        the full name (ie task_name + '_' + entry, where task_name is the name
        of the task that wrote the value in the database).
        """
        return self.task_database.get_value(self.task_path, full_name)

    def remove_from_database(self, full_name):
        """This method deletes the database entry specified by
        the full name (ie task_name + '_' + entry, where task_name is the name
        of the task that wrote the value in the database).
        """
        return self.task_database.delete_value(self.task_path, full_name)

    def register_in_database(self):
        """
        """
        if self.task_database_entries:
            for entry in self.task_database_entries:
                self.task_database.set_value(self.task_path,
                                             self.task_name + '_' + entry,
                                             self.task_database_entries[entry])

    def unregister_from_database(self):
        """
        """
        if self.task_database_entries:
            for entry in self.task_database_entries:
                self.task_database.delete_value(self.task_path,
                                                self.task_name + '_' + entry)

    def register_preferences(self):
        """
        """
        self.task_preferences.clear()
        for name in self.traits(preference = True):
            self.task_preferences[name] = str(self.get(name).values()[0])

    update_preferences_from_traits = register_preferences

    def update_traits_from_preferences(self, **parameters):
        """
        """
        for name, trait in self.traits(preference = True).iteritems():

            if not parameters.has_key(name):
                continue

            value = parameters[name]
            handler = trait.handler

            # If the trait type is 'Str' then we just take the raw value.
            if isinstance(handler, Str) or trait.is_str:
                pass

            # If the trait type is 'Unicode' then we convert the raw value.
            elif isinstance(handler, Unicode):
                value = unicode(value)

            # Otherwise, we eval it!
            else:
                try:
                    value = eval(value)

                # If the eval fails then there is probably a syntax error, but
                # we will let the handler validation throw the exception.
                except:
                    pass

            if handler.validate is not None:
                # Any traits have a validator of None.
                validated = handler.validate(self, name, value)
            else:
                validated = value

            self.trait_set(**{name : validated})

class ComplexTask(AbstractTask):
    """Task composed of several subtasks.
    """
    children_task = List(Instance(AbstractTask), child = True)
    has_root = Bool(False)

    def __init__(self, *args, **kwargs):
        super(ComplexTask, self).__init__(*args, **kwargs)
        self._define_task_view()
        self.on_trait_change(self._update_paths,
                             name = 'task_name, task_path, task_depth')

    @make_stoppable
    def process(self):
        """
        """
        for child in self.children_task:
            child.process()

    def check(self, *args, **kwargs):
        """Implementation of the test method of AbstractTask
        """
        test = True
        traceback = {}
        for name in self.traits(child = True):
            child = self.get(name).values()[0]
            if child:
                if isinstance(child, list):
                    for aux in child:
                        check = aux.check(*args, **kwargs)
                        test = test and check[0]
                        traceback.update(check[1])
                else:
                    check = child.check(*args, **kwargs)
                    test = test and check[0]
                    traceback.update(check[1])

        return test, traceback

    def create_child(self, ui):
        """Method to handle the adding of a child through the list editor
        """
        child = self.root_task.request_child(parent = self, ui = ui)
        return child

    def write_in_database(self, name, value):
        """This method build a task specific database entry from the task_name
        and the name arg and set the database entry to the value specified.
        """
        value_name = self.task_name + '_' + name
        return self.task_database.set_value(self.task_path, value_name, value)

    def get_from_database(self, full_name):
        """This method return the value under the database entry specified by
        the full name (ie task_name + '_' + entry, where task_name is the name
        of the task that wrote the value in the database).
        """
        return self.task_database.get_value(self.task_path, full_name)

    def remove_from_database(self, full_name):
        """This method deletes the database entry specified by
        the full name (ie task_name + '_' + entry, where task_name is the name
        of the task that wrote the value in the database).
        """
        return self.task_database.delete_value(self.task_path, full_name)

    def register_in_database(self):
        """
        """
        if self.task_database_entries:
            for entry in self.task_database_entries:
                self.task_database.set_value(self.task_path,
                                             self.task_name + '_' + entry,
                                             self.task_database_entries[entry])

        self.task_database.create_node(self.task_path, self.task_name)

        #ComplexTask defines children_task so we always get something
        for name in self.traits(child = True):
            child = self.get(name).values()[0]
            if child:
                if isinstance(child, list):
                    for aux in child:
                        aux.register_in_database()
                else:
                    child.register_in_database()

    def unregister_from_database(self):
        """
        """
        if self.task_database_entries:
            for entry in self.task_database_entries:
                self.task_database.delete_value(self.task_path,
                                                self.task_name + '_' + entry)

        for name in self.traits(child = True):
            child = self.get(name).values()[0]
            if child:
                if isinstance(child, list):
                    for aux in child:
                        aux.unregister_from_database()
                else:
                    child.unregister_from_database()

        self.task_database.delete_node(self.task_path, self.task_name)

    def register_preferences(self):
        """
        """
        self.task_preferences.clear()
        for name in self.traits(preference = True):
            self.task_preferences[name] = str(self.get(name).values()[0])

        for name in self.traits(child = True):
            child = self.get(name).values()[0]
            if child:
                if isinstance(child, list):
                    for i, aux in enumerate(child):
                        child_id = name + '_{}'.format(i)
                        self.task_preferences[child_id] = {}
                        aux.task_preferences = self.task_preferences[child_id]
                        aux.register_preferences()
                else:
                    self.task_preferences[name] = {}
                    child.task_preferences = self.task_preferences[name]
                    child.register_preferences()

    def update_preferences_from_traits(self):
        """
        """
        for name in self.traits(preference = True):
            self.task_preferences[name] = str(self.get(name).values()[0])

        for name in self.traits(child = True):
            child = self.get(name).values()[0]
            if child:
                if isinstance(child, list):
                    for aux in child:
                        aux.update_preferences_from_traits()
                else:
                    child.update_preferences_from_traits()

    def update_traits_from_preferences(self, **parameters):
        """

        NB : This method is fairly powerful and can handle a lot of cases so
        don't override it without checking that it works.
        """
        #First we set the preference traits
        for name, trait in self.traits(preference = True).iteritems():
            if not parameters.has_key(name):
                continue

            value = parameters[name]
            handler = trait.handler

            #if we get a container must check each member
            if isinstance(value, list):
                item_handler = handler.item_trait
                validated = []
                for string in value:
                    validated.append(self._from_str_to_basetrait(string,
                                                                 item_handler))
            if isinstance(value, dict):
                key_handler = handler.key_trait
                value_handler = handler.value_trait
                validated = {}
                for key, val in value.iteritems:
                    validated[self._from_str_to_basetrait(key, key_handler)] =\
                            self._from_str_to_basetrait(val, value_handler)

            #We have a standard value store as a string, we use the standard
            #procedure
            else:
                validated = self._from_str_to_basetrait(value, handler)

            self.trait_set(**{name : validated})

        #Then we set the child
        for name, trait in self.traits(child = True).iteritems():
            if not parameters.has_key(name):
                continue

            value = parameters[name]
            handler = trait.handler

            if isinstance(handler, List):
                validated = value
            else:
                validated = value[0]

            self.trait_set(**{name : validated})

    def _from_str_to_basetrait(self, value, trait):
        """
        """
         # If the trait type is 'Str' then we just take the raw value.
        if isinstance(trait, BaseStr) or trait.is_str:
            pass

        # If the trait type is 'Unicode' then we convert the raw value.
        elif isinstance(trait, BaseUnicode):
            value = unicode(value)

        # Otherwise, we eval it!
        else:
            value = eval(value)

        return value

    @on_trait_change('children_task[]')
    def on_children_modified(self, obj, name, old, new):
        """Handle children being added or removed from the task, no matter the
        source"""
        if self.has_root:
            if new and old:
                added = set(new) - set(old)
                removed = set(old) - set(new)
                if added:
                    for child in added:
                        self._child_added(child)
                if removed:
                    for child in removed:
                        self._child_removed(child)
            elif new:
                for child in new:
                    self._child_added(child)

            elif old:
                for child in old:
                    self._child_removed(child)


    #@on_trait_change('task_name, task_path, task_depth')
    def _update_paths(self, obj, name, old, new):
        """Method taking care that the path of children, the database and the
        task name remains coherent
        """
        if self.has_root:
            if name == 'task_name':
                self.task_database.rename_node(self.task_path, new, old)
                for name in self.traits(child = True):
                    child = self.get(name).values()[0]
                    if child:
                        if isinstance(child, list):
                            for aux in child:
                                aux.task_path = self.task_path + '/' + new
                        else:
                            child.task_path = self.task_path + '/' + new
            elif name == 'task_path':
                for name in self.traits(child = True):
                    child = self.get(name).values()[0]
                    if child:
                        if isinstance(child, list):
                            for aux in child:
                                aux.task_path = new + '/' + self.task_name
                        else:
                            child.task_path = new + '/' + self.task_name
            elif name == 'task_depth':
                for name in self.traits(child = True):
                    child = self.get(name).values()[0]
                    if child:
                        if isinstance(child, list):
                            for aux in child:
                                aux.task_depth = new + 1
                        else:
                            child.task_depth = new + 1


    def _child_added(self, child):
        """Method updating the database, depth and preference tree when a child
        is added
        """
        child.task_depth = self.task_depth + 1
        child.task_database = self.task_database
        child.task_path = self.task_path + '/' + self.task_name

        #Give him its root so that it can proceed to any child
        #registration it needs to.
        child.root_task = self.root_task

        #Ask the child to register in database
        child.register_in_database()
        #Register anew preferences to keep the right ordering for the childs
        self.register_preferences()

    def _child_removed(self, child):
        """Method updating the database and preference tree when a child is
        removed.
        """
        self.register_preferences()
        child.unregister_from_database()

    @on_trait_change('root_task')
    def _when_root(self, new):
        """Method making sure that all children get all the info they need to
        behave correctly when the task get its root parent (ie the task is now
        in a 'correct' environnement).
        """
        if new == None:
            return

        self.has_root = True
        for name in self.traits(child = True):
            child = self.get(name).values()[0]
            if child:
                if isinstance(child, list):
                    for aux in child:
                        aux.task_depth = self.task_depth + 1
                        aux.task_database = self.task_database
                        aux.task_path = self.task_path + '/' + self.task_name

                        #Give him its root so that it can proceed to any child
                        #registration it needs to.
                        aux.root_task = self.root_task
                else:
                    child.task_depth = self.task_depth + 1
                    child.task_database = self.task_database
                    child.task_path = self.task_path + '/' + self.task_name

                    #Give him its root so that it can proceed to any child
                    #registration it needs to.
                    child.root_task = self.root_task


    def _define_task_view(self):
        """
        """
        task_view = View(
                    VGroup(
                        UItem('task_name', style = 'readonly'),
                        HGroup(
                            UItem('children_task@',
                                  editor = ListInstanceEditor(
                                      style = 'custom',
                                      editor = InstanceEditor(view =
                                                              'task_view'),
                                      item_factory = self.create_child)),
                            show_border = True,
                            ),
                        ),
                        title = 'Edit task',
                        resizable = True,
                    )

        self.trait_view('task_view', task_view)

from multiprocessing.synchronize import Event

class RootTask(ComplexTask):
    """Special task which is always the root of a measurement and is the only
    task directly referencing the measurement editor.
    """
    default_path = Directory(preference = True)
    task_builder = Type()
    root_task = trait_self
    has_root = True
    task_database = TaskDatabase
    task_name = 'Root'
    task_label = 'Root'
    task_preferences = ConfigObj(indent_type = '    ')
    task_depth = 0
    task_path = 'root'
    task_database_entries = {'threads' : [], 'instrs' : {}, 'default_path' : ''}
    should_stop = Instance(Event)

    def __init__(self, *args, **kwargs):
        super(RootTask, self).__init__(*args, **kwargs)
        self.task_database = TaskDatabase()
        self.task_database.set_value('root', 'threads', [])
        self.task_database.set_value('root', 'instrs', {})
        self.task_database.set_value('root', 'default_path', '')

    def check(self, *args, **kwargs):
        traceback = {}
        test = True
        if not os.path.isdir(self.default_path):
            test = False
            traceback[self.task_path + '/' + self.task_name] =\
                'The provided default path is not a valid directory'
        check = super(RootTask, self).check(*args, **kwargs)
        test = test and check[0]
        traceback.update(check[1])
        return test, traceback

    @make_stoppable
    def process(self):
        """
        """
        for child in self.children_task:
            child.process()
        for thread in self.task_database.get_value('root','threads'):
            thread.join()
        instrs = self.task_database.get_value('root','instrs')
        for instr_profile in instrs:
            instrs[instr_profile].close_connection()

    def request_child(self, parent, ui):
        """
        """
        #the parent attribute is for now useless as all parent related traits
        #are set at adding time
        builder = self.task_builder()
        child = builder.build(parent = parent, ui = ui)
        return child

    def _child_added(self, child):
        """Method updating the database, depth and preference tree when a child
        is added
        """
        #Give the child all the info it needs to register
        child.task_depth = self.task_depth + 1
        child.task_database = self.task_database
        child.task_path = self.task_path

        #Give him its root so that it can proceed to any child
        #registration it needs to.
        child.root_task = self.root_task

        #Ask the child to register in database
        child.register_in_database()
        #Register anew preferences to keep the right ordering for the childs
        self.register_preferences()

    def _define_task_view(self):
        """
        """
        task_view = View(
                    VGroup(
                        UItem('task_name', style = 'readonly'),
                        UItem('default_path'),
                        HGroup(
                            UItem('children_task@',
                                  editor = ListInstanceEditor(
                                      style = 'custom',
                                      editor = InstanceEditor(view =
                                                              'task_view'),
                                      item_factory = self.create_child)),
                            show_border = True,
                            ),
                        ),
                        title = 'Edit task',
                        resizable = True,
                    )

        self.trait_view('task_view', task_view)

    def _task_class_default(self):
        return ComplexTask.__name__

    @on_trait_change('default_path')
    def _update_default_path_in_database(self, new):
        """
        """
        self.default_path = os.path.normpath(new)
        self.task_database.set_value('root', 'default_path', self.default_path)

AbstractTask.add_class_trait('root_task', Instance(RootTask))