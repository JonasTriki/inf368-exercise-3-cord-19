from pypatconsole import menu, clear_screen
from subprocess import Popen, call, DEVNULL
from time import sleep
import nvidia_smi

class decos:
    @staticmethod
    def waitinput(func):
        def wrapper(*args, **kwargs):
            ret_val = func(*args, **kwargs)
            print('\nPress ENTER to continue')
            input()
            return ret_val
        wrapper.__wrapped__ = func
        return wrapper

@decos.waitinput
def case1(targetfile: str = 'coordle/coordle_log.txt', size: int=None):
    '''
    Create and dump coordle index

    Shoul runs as zombie thread
    '''
    print(f'stdout will be routed to {targetfile}')
    print('Enter 1 to run indexing')
    confirm = input()
    if confirm == '1':
        Popen(f'nohup python coordle.py > {targetfile} &', shell=True)
    
if __name__ == '__main__':
    menu(locals(), blank_proceedure='pass', title=' Main menu ', main=True)
