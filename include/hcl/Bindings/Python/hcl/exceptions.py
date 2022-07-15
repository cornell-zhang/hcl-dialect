import warnings

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class HCLException(Exception):
    """Base class for all HeteroCL exceptions.

    Exception is the base class for warnings and errors.
    Developers can subclass this class to provide additional information

    Parameters
    ----------
    message : str
        The error message.
    """
    def __init__(self, message):
        Exception.__init__(self, message)


class HCLWarning(HCLException):
    """Base class for all HeteroCL warnings.

    Warning is the base class for all warnings.
    Developers can subclass this class to provide additional information

    Parameters
    ----------
    message : str
        The warning message.
    """
    def __init__(self, message):
        HCLException.__init__(self, message)

class HCLError(HCLException):
    """Base class for all HeteroCL errors.

    Error is the base class for all errors.
    Developers can subclass this class to provide additional information

    Parameters
    ----------
    message : str
        The error message.
    """
    def __init__(self, message):
        HCLException.__init__(self, message)