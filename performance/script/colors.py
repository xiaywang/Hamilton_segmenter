class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def green(s):
    return Colors.GREEN + s + Colors.END

def blue(s):
    return Colors.BLUE + s + Colors.END

def yellow(s):
    return Colors.YELLOW + s + Colors.END

def red(s):
    return Colors.RED + s + Colors.END

def title(s):
    return Colors.BOLD + Colors.BLUE + s +Colors.END

def bold(s):
    return Colors.BOLD + s + Colors.END

def underline(s):
    return Colors.UNDERLINE + s + Colors.END
