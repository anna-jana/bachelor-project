
def show(*args):
    """
    Prints out all of its arguments.
    Strings as text and floating point numbers in exponential form.
    """
    for arg in args:
        if isinstance(arg, str):
            print(arg, end=" ")
        else:
            print("%e" % arg, end=" ")
    print("")
