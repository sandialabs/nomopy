import numpy as np

class ConvergenceMonitor():
    def __init__(self, count, metric_fn, count_tol, metric_tol,
                    verbose=False):
        self.count = count
        self.metric_fn = metric_fn
        self.count_tol = count_tol
        self.metric_tol = metric_tol
        self.verbose = verbose

        self.metric = None
        self.metric_old = -np.inf

    def update(self):
        self.metric = self.metric_fn()
        if abs(self.metric - self.metric_old) <= self.metric_tol:
            self.count += 1
        else:
            self.count = 0
        self.metric_old = self.metric

    def update_has_converged(self):
        self.update()
        converged = self.count >= self.count_tol
        if converged and self.verbose:
            print("** CONVERGED!")
        return converged

    def reset(self):
        self.metric_old = -np.inf
