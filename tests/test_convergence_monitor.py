import numpy as np
import pytest

from nomopy.fhmm.convergence_monitor import ConvergenceMonitor


def test_simple_update_converging():
    scores = [5, 4.9, 4.8]
    metric_fn = iter(scores).__next__

    cm = ConvergenceMonitor(0, metric_fn, 5, 1)
    cm.update()
    assert cm.count == 0
    cm.update()
    assert cm.count == 1
    cm.update()
    assert cm.count == 2

def test_simple_update_not_converging():
    scores = [5, 4.9, 4.8]
    metric_fn = iter(scores).__next__

    cm = ConvergenceMonitor(0, metric_fn, 5, 0.01)
    cm.update()
    assert cm.count == 0
    cm.update()
    assert cm.count == 0
    cm.update()
    assert cm.count == 0

def test_simple_update_converging_then_not_converging():
    scores = [5, 4.9, 4.8, 10]
    metric_fn = iter(scores).__next__

    cm = ConvergenceMonitor(0, metric_fn, 5, 1)
    cm.update()
    assert cm.count==0
    cm.update()
    assert cm.count==1
    cm.update()
    assert cm.count==2
    cm.update()
    assert cm.count==0

def test_update_has_converged_converging():
    scores = [5, 4.9, 4.8]
    metric_fn = iter(scores).__next__

    cm = ConvergenceMonitor(0, metric_fn, 2, 1)
    converged = cm.update_has_converged()
    assert not converged
    converged = cm.update_has_converged()
    assert not converged
    converged = cm.update_has_converged()
    assert converged

def test_update_has_converged_not_converging():
    scores = [5, 4.9, 4.8]
    metric_fn = iter(scores).__next__

    cm = ConvergenceMonitor(0, metric_fn, 2, 0.01)
    converged = cm.update_has_converged()
    assert not converged
    converged = cm.update_has_converged()
    assert not converged
    converged = cm.update_has_converged()
    assert not converged


def test_convergence_monitor_reset():
    metric_fn = lambda : 5 # Always return 5

    cm = ConvergenceMonitor(0, metric_fn, 5, 1)
    cm.update()
    assert cm.metric_old == 5
    cm.reset()
    assert np.isneginf(cm.metric_old)
