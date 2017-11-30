import numpy as np

FEAT_KEY = 0
CUT_KEY = 1
LEFT_KEY = 2
RIGHT_KEY = 3
PRED_KEY = 4
DEPTH_KEY = 5
NUM_FIELDS = 6


class FlexArray:
    def __init__(self, num_fields, n=1, default_value=0, verbose=True):
        self.num_fields = num_fields
        self.default_value = default_value
        self.verbose = verbose
        self._data = np.ones((self.num_fields, n)) * default_value
        self._max_key = -1

    def __setitem__(self, field_key_tuple, value):
        field, key = field_key_tuple
        self._maybe_expand(key)
        self._data[field, key] = value

    def _maybe_expand(self, key):
        new_size = self._data.shape[1]
        if key > self._max_key:
            self._max_key = key
        while key >= new_size:
            new_size *= 2
        if self._data.shape[1] < new_size:
            if self.verbose:
                print("Expanding from {} to {} because of {}".format(
                    self._data.shape[1], new_size, key,
                ))

            self._data = np.hstack([
                self._data,
                np.ones((self.num_fields, new_size - self._data.shape[-1]))
                * self.default_value,
            ])

    def __getitem__(self, field_key_tuple):
        field, key = field_key_tuple
        return self._data[field, key]

    @property
    def max_key(self):
        return self._max_key

    @property
    def data(self):
        return self._data

    def get_column(self, field):
        return self._data[field]


class Tree(object):
    def __init__(self, k=None):
        self.k = k
        self.data_arr = None
        self.max_depth = 0
        self.length = 1
        self.fit_stack = None

    def fit(self, x, y):
        initial_size = int(np.ceil(np.log2(len(x))))
        if self.k is None:
            self.k = int(np.sqrt(x.shape[1]))
        self.data_arr = FlexArray(
            num_fields=NUM_FIELDS, n=initial_size, default_value=-1,
        )
        self.fit_stack = []
        self._fit_tree(
            x, y,
            prev_valid_feat_ls=range(x.shape[1]),
            pos=0,
            depth=0,
        )

    def _fit_tree(self, x, y, prev_valid_feat_ls, pos, depth):

        # Hack for python performance.
        self.data_arr[LEFT_KEY, pos] = pos
        self.data_arr[RIGHT_KEY, pos] = pos
        # End hack

        if depth > self.max_depth:
            self.max_depth = depth

        self.data_arr[DEPTH_KEY, pos] = depth
        if len(x) == 1:
            self.data_arr[PRED_KEY, pos] = y[0]
            return
        if len(np.unique(y)) == 1:
            self.data_arr[PRED_KEY, pos] = y[0]
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
            self.data_arr[PRED_KEY, pos] = np.random.choice(y)

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

        self.data_arr[FEAT_KEY, pos] = best_feat
        self.data_arr[CUT_KEY, pos] = best_cut

        left_pos = self.length
        right_pos = self.length + 1
        self.length += 2

        self._fit_tree(left_x, left_y, valid_feat_ls, pos=left_pos,
                       depth=depth + 1)
        self.data_arr[LEFT_KEY, pos] = left_pos

        self._fit_tree(right_x, right_y, valid_feat_ls, pos=right_pos,
                       depth=depth + 1)
        self.data_arr[RIGHT_KEY, pos] = right_pos

    def batch_predict(self, x):
        pos = self._batch_traverse_tree(x)
        return self.data_arr.data[PRED_KEY, pos].astype(int)

    def _batch_traverse_tree(self, x):
        x_range = np.arange(len(x))
        pos = np.zeros(len(x)).astype(int)
        depth_counter = 0

        while True:
            criteria = self.data_arr.data[:, pos]
            left_right = (
                x[x_range, criteria[FEAT_KEY].astype(int)] < criteria[CUT_KEY]
            )
            new_pos = np.zeros(len(x))
            new_pos[left_right] = \
                self.data_arr.data[LEFT_KEY, pos][left_right]
            new_pos[~left_right] = \
                self.data_arr.data[RIGHT_KEY, pos][~left_right]
            new_pos = new_pos.astype(int)
            if np.all(pos == new_pos):
                break
            pos = new_pos
            depth_counter += 1
            if depth_counter > 100000:
                raise Exception
        return pos

    def predict(self, x):
        return np.array([
            self.predict_with_tree(x_row)
            for x_row in x
        ])

    def predict_with_tree(self, x_row):
        feat_arr = self.data_arr.get_column(FEAT_KEY).astype(int)
        pos = 0
        while self.data_arr[PRED_KEY, pos] == -1:
            if x_row[feat_arr[pos]] < self.data_arr[CUT_KEY, pos]:
                pos = int(self.data_arr[LEFT_KEY, pos])
            else:
                pos = int(self.data_arr[RIGHT_KEY, pos])
        return self.data_arr[PRED_KEY, pos]
