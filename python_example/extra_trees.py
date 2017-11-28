import numpy as np


class FlexArray:
    def __init__(self, n=1, default_value=0):
        self._data = np.ones(n) * default_value
        self.default_value = default_value
        self._max_key = -1

    def __setitem__(self, key, value):
        new_size = len(self._data)
        if key > self._max_key:
            self._max_key = key
        while key >= new_size:
            new_size *= 2
        if len(self._data) < new_size:
            self._data = np.hstack([
                self._data,
                np.ones(new_size - len(self._data)) * self.default_value
            ])
        self._data[key] = value

    def __getitem__(self, key):
        return self._data[key]

    @property
    def max_key(self):
        return self._max_key

    @property
    def data(self):
        return self._data[:]


class Tree(object):
    def __init__(self, k=None):
        self.k = k
        self.feat_arr = None
        self.cut_arr = None
        self.pred_arr = None
        self.left_arr = None
        self.right_arr = None
        self.max_depth = 0
        self.length = 1

    def fit(self, x, y):
        initial_size = int(np.ceil(np.log2(len(x))))
        if self.k is None:
            self.k = int(np.sqrt(x.shape[1]))
        self.feat_arr = FlexArray(initial_size, default_value=-1)
        self.cut_arr = FlexArray(initial_size, default_value=-1)
        self.pred_arr = FlexArray(initial_size, default_value=-1)
        self.left_arr = FlexArray(initial_size, default_value=-1)
        self.right_arr = FlexArray(initial_size, default_value=-1)
        self._fit_tree(
            x, y,
            prev_valid_feat_ls=range(x.shape[1]),
            pos=0,
            depth=0,
        )

    def _fit_tree(self, x, y, prev_valid_feat_ls, pos, depth):
        if depth > self.max_depth:
            self.max_depth = depth
        if len(x) == 1:
            self.pred_arr[pos] = y[0]
            return
        if len(np.unique(y)) == 1:
            self.pred_arr[pos] = y[0]
            return

        valid_feat_min_max = []
        valid_feat_ls = []
        min_feat_x = np.min(x, axis=0)
        max_feat_x = np.max(x, axis=0)
        for feat in prev_valid_feat_ls:
            if min_feat_x[feat] != max_feat_x[feat]:
                valid_feat_min_max.append(
                    (feat, min_feat_x[feat], max_feat_x[feat]))
                valid_feat_ls.append(feat)

        if not valid_feat_ls:
            self.pred_arr[pos] = np.random.choice(y)

        chosen_feat = [
            valid_feat_min_max[_] for _ in
            np.random.choice(np.arange(len(valid_feat_ls)), replace=False,
                             size=min(self.k, len(valid_feat_ls)))
        ]

        cuts_ls = []
        best_improvement = -np.inf
        best_feat = None
        best_cut = None
        best_x_filtered = None
        for feat, feat_min, feat_max in chosen_feat:
            cut_point = np.random.uniform(feat_min, feat_max)
            x_filtered = x[:, feat] < cut_point
            cuts_ls.append(cut_point)

            _, counts_a = np.unique(y[x_filtered], return_counts=True)
            num_a = x_filtered.sum()
            _, counts_b = np.unique(y[~x_filtered], return_counts=True)
            num_b = len(x) - num_a
            impurity_a = 1 - ((counts_a / num_a) ** 2).sum()
            impurity_b = 1 - ((counts_b / num_b) ** 2).sum()
            proxy_improvement = (
                -num_a * impurity_a
                - num_b * impurity_b
            )
            if proxy_improvement > best_improvement:
                best_feat = feat
                best_cut = cut_point
                best_improvement = proxy_improvement
                best_x_filtered = x_filtered

        x_filtered = best_x_filtered
        left_x, left_y = x[x_filtered], y[x_filtered]
        right_x, right_y = x[~x_filtered], y[~x_filtered]

        self.feat_arr[pos] = best_feat
        self.cut_arr[pos] = best_cut

        left_pos = self.length
        right_pos = self.length + 1
        self.length += 2

        self._fit_tree(left_x, left_y, valid_feat_ls, pos=left_pos,
                       depth=depth + 1)
        self.left_arr[pos] = left_pos

        self._fit_tree(right_x, right_y, valid_feat_ls, pos=right_pos,
                       depth=depth + 1)
        self.right_arr[pos] = right_pos

    def predict(self, x):
        return np.array([
            self.predict_with_tree(x_row)
            for x_row in x
        ])

    def predict_with_tree(self, x_row):
        feat_arr = self.feat_arr.data.astype(int)
        pos = 0
        while self.pred_arr[pos] == -1:
            if x_row[feat_arr[pos]] < self.cut_arr[pos]:
                pos = int(self.left_arr[pos])
            else:
                pos = int(self.right_arr[pos])
        return self.pred_arr[pos]
