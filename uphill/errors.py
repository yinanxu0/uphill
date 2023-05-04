
class InitClassError(Exception):
    ...

class BadInputError(Exception):
    ...


'''
Common IO Errors
'''
class FileNotExistError(Exception):
    ...

class InvalidYAMLError(Exception):
    ...

class InvalidJsonError(Exception):
    ...

class InvalidJsonlError(Exception):
    ...

class BadFormatError(Exception):
    ...


'''
Audio Errors
'''
class AudioLoadingError(Exception):
    ...

class DurationMismatchError(Exception):
    ...

class NonPositiveEnergyError(ValueError):
    ...

