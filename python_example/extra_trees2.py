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
    def full_length(self):
        return self._data.shape[1]

    @property
    def data(self):
        return self._data

    def get_column(self, field):
        return self._data[field]


class Tree(object):
    def __init__(self, k=None):
        self.k = k
        self.data_arr = None
        self.length = 1
        self._x = None
        self._y = None

    def fit(self, x, y):
        initial_size = int(np.ceil(np.log2(len(x))))
        if self.k is None:
            self.k = int(np.sqrt(x.shape[1]))
        self.data_arr = FlexArray(
            num_fields=NUM_FIELDS, n=initial_size, default_value=-1,
        )

        self.data_arr[LEFT_KEY, 0] = 0
        self.data_arr[RIGHT_KEY, 0] = 0
        self.data_arr[DEPTH_KEY, 0] = 0

        self._x = x
        self._y = y

        curr_depth = 0
        while np.any(self.data_arr.get_column(DEPTH_KEY) == curr_depth):
            batch_pos = self._batch_traverse_tree(self._x)
            pos_candidates = np.arange(self.data_arr.full_length)[
                self.data_arr.get_column(DEPTH_KEY) == curr_depth
            ]
            for pos in pos_candidates:
                node_filter = batch_pos == pos
                self._fit_tree(node_filter, pos)
            curr_depth += 1

    def _fit_tree(self, node_filter, pos):
        x = self._x[node_filter]
        y = self._y[node_filter]

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
        for feat in np.arange(x.shape[1]):
            if min_feat_x[feat] != max_feat_x[feat]:
                valid_feat_min_max.append(
                    (feat, min_feat_x[feat], max_feat_x[feat]))
                valid_feat_ls.append(feat)

        if not valid_feat_ls:
            self.data_arr[PRED_KEY, pos] = np.random.choice(y)

        chosen_feat = [
            valid_feat_min_max[_] for _ in
            np.random.choice(np.arange(len(valid_feat_ls)), replace=True,
                             size=min(self.k, len(valid_feat_ls)))
        ]

        cuts_ls = []
        best_improvement = -np.inf
        best_feat = None
        best_cut = None
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

        self.data_arr[FEAT_KEY, pos] = best_feat
        self.data_arr[CUT_KEY, pos] = best_cut

        left_pos = self.length
        right_pos = self.length + 1
        self.length += 2

        self.data_arr[LEFT_KEY, pos] = left_pos
        self.data_arr[RIGHT_KEY, pos] = right_pos

        # Hack for python performance to terminate traversal
        depth = self.data_arr[DEPTH_KEY, pos]
        self.data_arr[LEFT_KEY, left_pos] = left_pos
        self.data_arr[DEPTH_KEY, left_pos] = depth + 1
        self.data_arr[RIGHT_KEY, left_pos] = left_pos

        self.data_arr[LEFT_KEY, right_pos] = right_pos
        self.data_arr[RIGHT_KEY, right_pos] = right_pos
        self.data_arr[DEPTH_KEY, right_pos] = depth + 1
        # End hack

    def predict(self, x):
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
