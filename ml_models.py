import numpy as np
import scipy.stats as stats

class StandardScalerCustom:
    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, X):
        X = np.array(X, dtype=np.float64)
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        # Avoid division by zero
        self.scale[self.scale == 0] = 1.0
        return self

    def transform(self, X):
        X = np.array(X, dtype=np.float64)
        return (X - self.mean) / self.scale

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.array(X, dtype=np.float64)
        return X * self.scale + self.mean


class PowerTransformerCustom:
    def __init__(self):
        self.lambdas = []

    def fit(self, X):
        X = np.array(X, dtype=np.float64)
        self.lambdas = []
        n_features = X.shape[1]
        for i in range(n_features):
            col = X[:, i]
            # Use scipy.stats.yeojohnson to find the optimal lambda
            _, lmbda = stats.yeojohnson(col)
            self.lambdas.append(lmbda)
        return self

    def transform(self, X):
        X = np.array(X, dtype=np.float64)
        X_trans = np.empty_like(X)
        for i in range(X.shape[1]):
            X_trans[:, i] = stats.yeojohnson(X[:, i], lmbda=self.lambdas[i])
        return X_trans

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# Custom Train-Test Split (reproducing split logic)
def train_test_split_custom(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    n_test = int(np.round(n_samples * test_size))
    if n_test == 0:
        n_test = 1
    elif n_test == n_samples:
        n_test = n_samples - 1
        
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    if isinstance(X, np.ndarray):
        X_train, X_test = X[train_indices], X[test_indices]
    else:
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        
    if isinstance(y, np.ndarray):
        y_train, y_test = y[train_indices], y[test_indices]
    else:
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        
    return X_train, X_test, y_train, y_test


# Regressors
class LinearRegressionCustom:
    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()
        # Add bias term
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        self.w = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_b @ self.w


class RidgeRegressionCustom:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        A = X_b.T @ X_b + self.alpha * np.eye(X_b.shape[1])
        A[0, 0] = (X_b.T @ X_b)[0, 0] # Don't regularize intercept
        self.w = np.linalg.pinv(A) @ X_b.T @ y
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        X_b = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_b @ self.w


class LassoRegressionCustom:
    def __init__(self, alpha=0.1, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.intercept = np.mean(y)
        
        # Coordinate descent
        y_centered = y - self.intercept
        for _ in range(self.max_iter):
            w_old = self.w.copy()
            for j in range(n_features):
                # Calculate residual without feature j
                r = y_centered - (X @ self.w - X[:, j] * self.w[j])
                rho = X[:, j] @ r
                
                # Soft thresholding
                norm_x = np.sum(X[:, j]**2)
                if norm_x == 0:
                    self.w[j] = 0
                else:
                    if rho < -self.alpha:
                        self.w[j] = (rho + self.alpha) / norm_x
                    elif rho > self.alpha:
                        self.w[j] = (rho - self.alpha) / norm_x
                    else:
                        self.w[j] = 0
            
            if np.sum(np.abs(self.w - w_old)) < self.tol:
                break
        
        self.intercept = np.mean(y - X @ self.w)
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        return X @ self.w + self.intercept


class ElasticNetRegressionCustom:
    def __init__(self, alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.intercept = np.mean(y)
        
        # Coordinate descent for ElasticNet
        y_centered = y - self.intercept
        for _ in range(self.max_iter):
            w_old = self.w.copy()
            for j in range(n_features):
                r = y_centered - (X @ self.w - X[:, j] * self.w[j])
                rho = X[:, j] @ r
                
                norm_x = np.sum(X[:, j]**2)
                if norm_x == 0:
                    self.w[j] = 0
                else:
                    # ElasticNet penalty updates
                    l1_pen = self.alpha * self.l1_ratio
                    l2_pen = self.alpha * (1.0 - self.l1_ratio)
                    
                    if rho < -l1_pen:
                        self.w[j] = (rho + l1_pen) / (norm_x + l2_pen)
                    elif rho > l1_pen:
                        self.w[j] = (rho - l1_pen) / (norm_x + l2_pen)
                    else:
                        self.w[j] = 0
            
            if np.sum(np.abs(self.w - w_old)) < self.tol:
                break
        
        self.intercept = np.mean(y - X @ self.w)
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        return X @ self.w + self.intercept


# Decision Tree node
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTreeRegressorCustom:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()
        self.root = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if (depth >= self.max_depth or n_samples < self.min_samples_split or np.std(y) == 0):
            return Node(value=np.mean(y))

        # Find best split
        best_feat, best_thresh = None, None
        best_mse = np.var(y)
        
        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                y_l, y_r = y[left_mask], y[right_mask]
                mse = (len(y_l)*np.var(y_l) + len(y_r)*np.var(y_r)) / n_samples
                if mse < best_mse:
                    best_mse = mse
                    best_feat = feat
                    best_thresh = thresh

        if best_feat is None:
            return Node(value=np.mean(y))

        left_mask = X[:, best_feat] <= best_thresh
        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature=best_feat, threshold=best_thresh, left=left_node, right=right_node)

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        return np.array([self._predict_row(self.root, row) for row in X])

    def _predict_row(self, node, row):
        if node.is_leaf():
            return node.value
        if row[node.feature] <= node.threshold:
            return self._predict_row(node.left, row)
        return self._predict_row(node.right, row)


class RandomForestRegressorCustom:
    def __init__(self, n_estimators=10, max_depth=3, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()
        np.random.seed(self.random_state)
        self.trees = []
        n_samples = X.shape[0]
        
        for _ in range(self.n_estimators):
            # Bootstrap sample
            boot_idx = np.random.choice(n_samples, n_samples, replace=True)
            tree = DecisionTreeRegressorCustom(max_depth=self.max_depth)
            tree.fit(X[boot_idx], y[boot_idx])
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(preds, axis=0)


class ExtraTreesRegressorCustom:
    # ExtraTrees is similar to Random Forests but selects splits completely at random
    def __init__(self, n_estimators=10, max_depth=3, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()
        np.random.seed(self.random_state)
        self.trees = []
        n_samples, n_features = X.shape
        
        for _ in range(self.n_estimators):
            # Boostrap sample
            boot_idx = np.random.choice(n_samples, n_samples, replace=True)
            # Create a randomized tree (we select split points randomly)
            tree = self._build_random_tree(X[boot_idx], y[boot_idx], depth=0)
            self.trees.append(tree)
        return self

    def _build_random_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        if (depth >= self.max_depth or n_samples < 2 or np.std(y) == 0):
            return Node(value=np.mean(y))

        # Select a subset of features and a random split point for each
        best_feat, best_thresh = None, None
        best_mse = np.var(y)
        
        # Select random subset of features
        n_sub = max(1, int(np.sqrt(n_features)))
        feats = np.random.choice(n_features, n_sub, replace=False)
        
        for feat in feats:
            # Pick a random threshold between min and max
            val_min, val_max = np.min(X[:, feat]), np.max(X[:, feat])
            if val_min == val_max:
                continue
            thresh = np.random.uniform(val_min, val_max)
            
            left_mask = X[:, feat] <= thresh
            right_mask = ~left_mask
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            y_l, y_r = y[left_mask], y[right_mask]
            mse = (len(y_l)*np.var(y_l) + len(y_r)*np.var(y_r)) / n_samples
            if mse < best_mse:
                best_mse = mse
                best_feat = feat
                best_thresh = thresh

        if best_feat is None:
            return Node(value=np.mean(y))

        left_mask = X[:, best_feat] <= best_thresh
        left_node = self._build_random_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_random_tree(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature=best_feat, threshold=best_thresh, left=left_node, right=right_node)

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        preds = []
        for tree in self.trees:
            pred = np.array([self._predict_row(tree, row) for row in X])
            preds.append(pred)
        return np.mean(preds, axis=0)

    def _predict_row(self, node, row):
        if node.is_leaf():
            return node.value
        if row[node.feature] <= node.threshold:
            return self._predict_row(node.left, row)
        return self._predict_row(node.right, row)


class GradientBoostingRegressorCustom:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=2, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.initial_pred = None

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()
        self.initial_pred = np.mean(y)
        y_pred = np.full_like(y, self.initial_pred, dtype=np.float64)
        self.trees = []
        
        for _ in range(self.n_estimators):
            residuals = y - y_pred
            tree = DecisionTreeRegressorCustom(max_depth=self.max_depth)
            tree.fit(X, residuals)
            y_pred += self.learning_rate * tree.predict(X)
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        preds = np.full(X.shape[0], self.initial_pred, dtype=np.float64)
        for tree in self.trees:
            preds += self.learning_rate * tree.predict(X)
        return preds


class AdaBoostRegressorCustom:
    # AdaBoost.R2 algorithm
    def __init__(self, n_estimators=10, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators = []
        self.estimator_weights = []

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()
        n_samples = X.shape[0]
        np.random.seed(self.random_state)
        
        # Initialize weights
        w = np.ones(n_samples) / n_samples
        self.estimators = []
        self.estimator_weights = []
        
        for _ in range(self.n_estimators):
            # Fit weak learner
            tree = DecisionTreeRegressorCustom(max_depth=2)
            # Weighted bootstrap sampling
            sample_idx = np.random.choice(n_samples, n_samples, replace=True, p=w)
            tree.fit(X[sample_idx], y[sample_idx])
            
            y_pred = tree.predict(X)
            errors = np.abs(y_pred - y)
            max_error = np.max(errors)
            if max_error == 0:
                max_error = 1.0
            errors /= max_error  # scale to [0, 1]
            
            # Weighted error rate
            err_rate = np.sum(w * errors)
            if err_rate >= 0.5:
                break
                
            beta = err_rate / (1.0 - err_rate)
            if beta == 0:
                beta = 1e-10
                
            # Update sample weights
            w = w * (beta ** (1.0 - errors))
            w_sum = np.sum(w)
            if w_sum == 0:
                break
            w /= w_sum
            
            self.estimators.append(tree)
            self.estimator_weights.append(np.log(1.0 / beta))
            
        if len(self.estimators) == 0:
            # Fallback
            fallback = DecisionTreeRegressorCustom(max_depth=1)
            fallback.fit(X, y)
            self.estimators.append(fallback)
            self.estimator_weights.append(1.0)
            
        self.estimator_weights = np.array(self.estimator_weights)
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        # Prediction is the weighted median
        preds = np.array([tree.predict(X) for tree in self.estimators])
        
        # Sort predictions for each sample and find weighted median
        final_preds = []
        for i in range(X.shape[0]):
            sample_preds = preds[:, i]
            sort_idx = np.argsort(sample_preds)
            sorted_preds = sample_preds[sort_idx]
            sorted_weights = self.estimator_weights[sort_idx]
            
            cum_weights = np.cumsum(sorted_weights)
            threshold = 0.5 * np.sum(sorted_weights)
            median_idx = np.where(cum_weights >= threshold)[0][0]
            final_preds.append(sorted_preds[median_idx])
            
        return np.array(final_preds)


class KNeighborsRegressorCustom:
    def __init__(self, n_neighbors=2):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y, dtype=np.float64).ravel()
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        preds = []
        for row in X:
            dists = np.sqrt(np.sum((self.X_train - row)**2, axis=1))
            nearest_idx = np.argsort(dists)[:self.n_neighbors]
            preds.append(np.mean(self.y_train[nearest_idx]))
        return np.array(preds)


class SVRCustom:
    # A simple gradient-descent support vector regression (with linear kernel)
    def __init__(self, C=1.0, epsilon=0.1, lr=0.01, epochs=500):
        self.C = C
        self.epsilon = epsilon
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        for _ in range(self.epochs):
            preds = X @ self.w + self.b
            diffs = preds - y
            
            dw = self.w.copy()
            db = 0.0
            
            for i in range(n_samples):
                d = diffs[i]
                if d > self.epsilon:
                    dw += self.C * X[i]
                    db += self.C
                elif d < -self.epsilon:
                    dw -= self.C * X[i]
                    db -= self.C
                    
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        return X @ self.w + self.b
