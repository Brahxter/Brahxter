import copy


class MarketTask:
    def __init__(self, batch_data, pattern_type, split_ratio=0.8):
        """
        batch_data: a dict containing at least {'prices': ..., 'target': ...}
                    where the values are tensors of shape (seq, features) 
                    (for a single trading day) or (batch, seq, features).
        pattern_type: a string indicating the task type (e.g., 'trend_following', 'mean_reversion', 'breakout')
        split_ratio: the fraction of the sequence to use for the support set; the remainder is for the query.
        """
        self.data = batch_data
        self.pattern_type = pattern_type
        self.split_ratio = split_ratio
        # Determine split point based on the sequence length of prices (assumed to be at dimension 0)
        seq_len = self.data['prices'].shape[0]
        self.split_point = int(seq_len * self.split_ratio)

    def get_support_set(self):
        # Returns a dict with the support set slices for each key in data.
        support = {}
        for key, value in self.data.items():
            support[key] = value[:self.split_point]
        return support

    def get_query_set(self):
        # Returns the query portion of the data.
        query = {}
        for key, value in self.data.items():
            query[key] = value[self.split_point:]
        return query
