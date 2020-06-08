import mir_eval
import numpy as np

def evaluate(ref, est):
    ref_itv = ref[:,:2] # extract the first two columns
    ref_pitch = ref[:, 2]

    est_itv = est[:, :2]
    est_pitch = est[:, 2]

    score = mir_eval.transcription.evaluate(ref_itv, ref_pitch, est_itv, est_pitch) 
    f_all = score['F-measure']
    f_noOff = score['F-measure_no_offset']
    f_onset = score['Onset_F-measure']
    print(f_all, f_noOff, f_onset)
    return f_all * 0.2 + f_noOff * 0.6 + f_onset * 0.2