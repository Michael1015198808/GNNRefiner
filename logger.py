from time import ctime, time

def log(*args, **kwargs):
    print(ctime(), *args, **kwargs)

class Timer:
    def __init__(self):
        self.t = time()
    def log(self, *args, **kwargs):
        print(time() - self.t, *args, **kwargs)
        self.t = time()
