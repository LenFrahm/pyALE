import timeit

def tic():
    return timeit.default_timer()

def toc(starttime):
    return timeit.default_timer() - starttime