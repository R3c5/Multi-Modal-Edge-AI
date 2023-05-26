class HyperparameterConfig:
    def __init__(self, window_size: float = 10, window_slide: float = 5, event_based=True, one_hot=True):
        self.window_size = window_size
        self.window_slide = window_slide
        self.event_based = event_based
        self.one_hot = one_hot
