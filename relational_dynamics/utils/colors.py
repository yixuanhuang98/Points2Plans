import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

    Black = '\033[30m'
    Red = '\033[31m'
    Green = '\033[32m'
    Yellow = '\033[33m'
    Blue = '\033[34m'
    Purple = '\033[35m'
    Cyan = '\033[36m'
    LightGray = '\033[37m'
    DarkGray = '\033[30m'
    LightRed = '\033[31m'
    LightGreen = '\033[32m'
    LightYellow = '\033[93m'
    LightBlue = '\033[34m'
    LightPurple = '\033[35m'
    LightCyan = '\033[36m'
    White = '\033[97m'

    BckgrDefault = '\033[49m'
    BckgrBlack = '\033[40m'
    BckgrRed = '\033[41m'
    BckgrGreen = '\033[42m'
    BckgrYellow = '\033[43m'
    BckgrBlue = '\033[44m'
    BckgrPurple = '\033[45m'
    BckgrCyan = '\033[46m'
    BckgrLightGray = '\033[47m'
    BckgrDarkGray = '\033[100m'
    BckgrLightRed = '\033[101m'
    BckgrLightGreen = '\033[102m'
    BckgrLightYellow = '\033[103m'
    BckgrLightBlue = '\033[104m'
    BckgrLightPurple = '\033[105m'


    @staticmethod
    def header(msg):
        return bcolors.HEADER + msg + bcolors.ENDC

    @staticmethod
    def okblue(msg):
        return bcolors.OKBLUE + msg + bcolors.ENDC

    @staticmethod
    def okgreen(msg):
        return bcolors.OKGREEN + msg + bcolors.ENDC

    @staticmethod
    def warning(msg):
        return bcolors.WARNING + msg + bcolors.ENDC

    @staticmethod
    def fail(msg):
        return bcolors.FAIL + msg + bcolors.ENDC

    @staticmethod
    def c_cyan(msg):
        return bcolors.Cyan + msg + bcolors.ENDC

    @staticmethod
    def c_red(msg):
        return bcolors.Red + msg + bcolors.ENDC

    @staticmethod
    def c_yellow(msg):
        return bcolors.Yellow + msg + bcolors.ENDC

    @staticmethod
    def c_blue(msg):
        return bcolors.Blue + msg + bcolors.ENDC

    @staticmethod
    def c_purple(msg):
        return bcolors.Purple + msg + bcolors.ENDC

    @staticmethod
    def c_green(msg):
        return bcolors.Green + msg + bcolors.ENDC
