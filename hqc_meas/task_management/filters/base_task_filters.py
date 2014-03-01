# -*- coding: utf-8 -*-
#==============================================================================
# module : base_task_filters.py
# author : Matthieu Dartiailh
# license : MIT license
#==============================================================================
""" Modules defining the basic filters.

:Contains:
    AbstractTaskFilter:
        Base class defining the interface.
    AllTaskFilter:
    PyTaskFilter:
    TemplateTaskFilter:
    SubclassFilter:
        Tool class for filter using issubclass.
    SimpleTaskFilter:
    ComplexTaskFilter:
    LoopTaskFilter:
    InstrumentTaskFilter:
    LoopableTaskFilter:
    TASK_FILTERS:
        Dict mapping useful filters to their names.

"""
from ...tasks.api import (SimpleTask, ComplexTask, BaseLoopTask,
                          InstrumentTask)


class AbstractTaskFilter(object):
    """ Base class for all task filters.

    Filters should simply override the filter_tasks classmethod.

    """

    @classmethod
    def filter_tasks(cls, py_tasks, template_tasks):
        """ Class method used to filter tasks.

        Parameters
        ----------
            py_tasks : dict
                Dictionary of known python tasks as name : class

            template_tasks : dict
                Dictionary of known templates as name : path

        Returns
        -------
            task_names : list(str)
                List of the name of the task matching the filters criteria.

        """
        err_str = 'This method should be implemented by subclasses of\
        AbstractTaskFilter. This method is called when the program requires\
        the task filter to filter the list of available tasks'
        raise NotImplementedError(err_str)


class AllTaskFilter(AbstractTaskFilter):
    """ Filter returning all tasks.

    """

    @classmethod
    def filter_tasks(cls, py_tasks, template_tasks):

        return list(py_tasks.keys()) + list(template_tasks.keys())


class PyTaskFilter(AbstractTaskFilter):
    """ Filter keeping only the python tasks.

    """

    @classmethod
    def filter_tasks(cls, py_tasks, template_tasks):

        return py_tasks.keys()


class TemplateTaskFilter(AbstractTaskFilter):
    """ Filter keeping only the templates.

    """

    @classmethod
    def filter_tasks(cls, py_tasks, template_tasks):

        return template_tasks.keys()


class SubclassFilter(AbstractTaskFilter):
    """ Filter keeping only the python tasks which are subclass of task_class.

    """

    # Class attribute to which task will be compared.
    task_class = type

    def filter_tasks(cls, py_tasks, template_tasks):
        """
        """
        tasks = []
        for name, t_class in py_tasks.iteritems():
            if issubclass(t_class, cls.task_class):
                tasks.append(name)

        return tasks


class SimpleTaskFilter(SubclassFilter):
    """ Filter keeping only the subclasses of SimpleTask.

    """
    task_class = SimpleTask


class ComplexTaskFilter(SubclassFilter):
    """ Filter keeping only the subclasses of ComplexTask.

    """
    task_class = ComplexTask


class LoopTaskFilter(SubclassFilter):
    """ Filter keeping only the subclasses of BaseLoopTask.

    """
    task_class = BaseLoopTask


class InstrumentTaskFilter(AbstractTaskFilter):
    """ Filter keeping only the subclasses of InstrumentTask.

    """
    task_class = InstrumentTask


class LoopableTaskFilter(AbstractTaskFilter):
    """ Filter keeping only the task declared to be loopable.

    """

    def filter_tasks(cls, py_tasks, template_tasks):
        """
        """
        tasks = []
        for name, t_class in py_tasks.iteritems():
            if hasattr(t_class, 'loopable') and t_class.loopable:
                tasks.append(name)

        return tasks

TASK_FILTERS = {'All': AllTaskFilter,
                'Python': PyTaskFilter,
                'Template': TemplateTaskFilter,
                'Simple': SimpleTaskFilter,
                'Complex': ComplexTaskFilter,
                'Loopable': LoopableTaskFilter,
                'Instrs': InstrumentTaskFilter,
                'Loop': LoopTaskFilter}
