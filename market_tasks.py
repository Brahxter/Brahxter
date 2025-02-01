class MarketTask:
    def __init__(self, data, pattern_type):
        self.data = data
        self.pattern_type = pattern_type  # e.g., 'trend_following', 'mean_reversion'
        
    def get_support_set(self):
        # Historical examples of this pattern
        return self.data[:self.split_point]
        
    def get_query_set(self):
        # New instances to test adaptation
        return self.data[self.split_point:]
