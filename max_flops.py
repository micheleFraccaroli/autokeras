class MaxFlops:
    def __init__(self, max_flops):
        self.max_flops = max_flops

    def set(self, max):
        self.max_flops = max

    def get(self):
        return self.max_flops