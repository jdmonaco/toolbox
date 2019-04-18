"""
Functions for supporting parallel computations.
"""

from pouty import log, printf


status_keys = ('completed', 'queue', 'tasks')


def total_complete(view):
    tot = 0
    for eid, status in view.queue_status().items():
        if type(status) is not dict:
            continue
        tot += status['completed']
    return tot

def queue_status_progress_bar(view, timeout=10):
    """Output a progress bar for a view's queue."""
    previous = total_complete(view)
    while not view.wait(timeout=timeout):
        current = total_complete(view)
        new = current - previous
        previous = current
        printf('\u25a0' * new)
    printf('\u25a0' * (total_complete(view) - previous))
    printf('\n')

def queue_status_progress_message(view, timeout=60):
    """Output a full status message for a view's queue."""
    while not view.wait(timeout=timeout):
        msg = ['Engine status:']
        qs = view.queue_status()
        for eid in view.client.ids:
            tasks = ', '.join(['{} {}'.format(qs[eid][k], k)
                for k in status_keys])
            msg.append('Engine {}: {}'.format(eid, tasks))
        log('\n'.join(msg))
    log('Queue completed')
