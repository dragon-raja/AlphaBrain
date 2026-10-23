"""Injectable monotonic phase timings; no GPU synchronization or scientific changes."""
from contextlib import contextmanager
import time


class PhaseTiming:
    def __init__(self,clock=time.perf_counter):self.clock=clock;self.rows={}

    @contextmanager
    def measure(self,name):
        begin=self.clock()
        try:yield
        finally:
            row=self.rows.setdefault(name,dict(seconds=0.,count=0))
            row['seconds']+=self.clock()-begin;row['count']+=1

    @contextmanager
    def context(self,context,name):
        began=self.clock();closing=None
        try:
            with context as value:
                self.rows.setdefault(name+'_start',dict(seconds=0.,count=0))
                self.rows[name+'_start']['seconds']+=self.clock()-began
                self.rows[name+'_start']['count']+=1
                try:yield value
                finally:closing=self.clock()
        finally:
            if closing is not None:
                row=self.rows.setdefault(name+'_close',dict(seconds=0.,count=0))
                row['seconds']+=self.clock()-closing;row['count']+=1
