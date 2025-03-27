import pytest
import numpy as np
import pandas as pd
import copy

from pyMassEvac import mutable_priority_queue

def test_init():
    """Test the init method.

    Test the init method and confirm that it returns a data frame with two
    columns and zero rows.
    """

    q = mutable_priority_queue.MutablePriorityQueue()

    assert q.queue.shape == (0, 2)

def test_put():
    """Test the put method.

    Test the put method and confirm that it inserts an event into the queue.
    """

    q = mutable_priority_queue.MutablePriorityQueue()
    original_shape = q.queue.shape

    q.put(tau_k = 12, e_k = 2, setRelative = True)
    
    assert original_shape[0] + 1 == q.queue.shape[0]

def test_get():
    """Test the get method.

    Test the get method and confirm that it removes an event into the queue.
    """

    q = mutable_priority_queue.MutablePriorityQueue()
    q.put(tau_k = 12, e_k = 2, setRelative = True)
    original_shape = q.queue.shape

    q.get()

    assert original_shape[0] - 1 == q.queue.shape[0]
