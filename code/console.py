from pypatconsole import menu, clear_screen
from subprocess import Popen, call, DEVNULL
from time import sleep
import nvidia_smi

class decorators:
    @staticmethod
    def waitinput(func):
        def wrapper(*args, **kwargs):
            ret_val = func(*args, **kwargs)
            print('\nPress ENTER to continue')
            input()
            return ret_val
        wrapper.__wrapped__ = func
        return wrapper

@decorators.waitinput
def case1():
    '''
    List directory
    
    Just to try something
    '''
    call('ls -al', shell=True)

@decorators.waitinput
def case2():
    '''
    Neofetch!
    
    Just to try something
    '''
    call('neofetch', shell=True)

@decorators.waitinput
def case3():
    '''
    Get GPU status
    '''
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    mem_res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(f'mem: {mem_res.used / (1024**2)} (MiB) {100 * (mem_res.used / mem_res.total):.3f}%') 
    
if __name__ == '__main__':
    menu(locals(), blank_proceedure='pass', title=' Main menu ', main=True)