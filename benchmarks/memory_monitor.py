import resource
import time

class MemoryMonitor:
    def __init__(self):
        self.keep_measuring = True

    def measure_usage(self):
        usage = []
        while self.keep_measuring:
            usage.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            time.sleep(0.1)

        return usage
