"""Metrics for adaptive runs, computed from the per-query rows of a replay (semwisp.sim.Trace.df)."""
import numpy as np


def balanced_accuracy(df):
    """Mean of person recall and empty recall of the server's answers, over queries with a known label."""
    d = df[df.label >= 0]
    return float((d.answer[d.label == 1].mean() + (1 - d.answer[d.label == 0]).mean()) / 2)


def summary(df):
    out = dict(queries=len(df), balanced_accuracy=balanced_accuracy(df), full_frames=int(df.explored.sum()),
               full_frame_rate=float(df.explored.mean()), up_bytes_per_query=float(df.up_bytes.mean()),
               reconfigs=int(df.reconfig.sum()), down_bytes=float(df.down_bits.sum() / 8))
    if 'thumbed' in df:                              # TieredServer: share of queries with a thumbnail check
        out['thumb_rate'] = float(df.thumbed.mean())
    return out


def readaptation(df):
    """One stretch of queries that starts with a change (a new room, or the first power-up).

    detect_queries   queries until the server noticed the change and went back to full frames (None if it never
                     did; 0 for a cold start, where it begins on full frames anyway)
    deploy_queries   queries until the tag was running a code learned after the change
    full_frames_to_deploy   full frames fetched up to that point: the price of readapting
    then `after`: balanced accuracy, full-frame rate and bytes per query from deployment to the end."""
    n = len(df)
    cold = not bool(df.used_code.iloc[0])
    changed = np.flatnonzero(df.changed.values) if 'changed' in df else np.array([], int)
    detect = 0 if cold else int(changed[0]) if len(changed) else None
    start = detect if detect is not None else 0
    reconf = np.flatnonzero(df.reconfig.values)
    reconf = reconf[reconf >= start]
    if len(reconf) == 0:
        return dict(detect_queries=detect, deploy_queries=None, full_frames_to_deploy=None, after=None,
                    whole=summary(df))
    deploy = int(reconf[0])
    return dict(detect_queries=detect, deploy_queries=deploy, full_frames_to_deploy=int(df.explored.values[:deploy].sum()),
                after=summary(df.iloc[deploy:]) if n - deploy >= 50 else None, whole=summary(df))
