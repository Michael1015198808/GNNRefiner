from time import ctime

def log(*args, **kwargs):
    print(ctime(), *args, **kwargs)
