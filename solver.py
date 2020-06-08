import numpy as np

FRAME_SIZE = 0.016
THRESHOLD = 15

def trim_data(pitch_set):
    trimmed_set = []
    new_set = []
    last_zero = True
    for pair in pitch_set:
        if pair[1] > 0:
            new_set.append(pair)
            last_zero = False
        elif last_zero == False:
            trimmed_set.append(new_set)
            new_set = []
            last_zero = True
    return trimmed_set

def solve(pitch_set):
    data_len = len(pitch_set)
    err_table = []

    # init the table O(N)
    first_row = [0 for _ in range(data_len)]
    note = np.array([pitch_set[0][1]])
    for i in range(1,data_len):
        note = np.append(note, pitch_set[i][1])
        first_row[i] = np.sum(np.abs(note - np.median(note)))
    err_table.append(first_row)

    predict_note = int(round(data_len * 1))
    
    '''
    print(f'data_len: {data_len}, predict_note: {predict_note}')
    print("finish initializing")
    '''
    
    # fill the table O(NM)
    for note_idx in range(1,predict_note):
        #print(f'index {note_idx}')
        row = [0 for _ in range(data_len)]
        cur_note = np.array([0])
        row[note_idx-1] = 100000 # a large number
        cnt = 1
        for i in range(note_idx+1,data_len):
            temp = np.append(cur_note, pitch_set[i][1])
            can1 = np.sum(np.abs(temp - np.median(temp)))
            can2 = err_table[note_idx-1][i-1]
            if can1 == can2:
                if cnt < 3:
                    cur_note = temp
                    row[i] = can1 + 0.01
                    cnt += 1
                else:
                    row[i] = can2
                    cnt = 1
            elif can1 < can2:
                if cnt > 8 and abs(can1-can2) < THRESHOLD:
                    row[i] = can2
                    cnt = 1
                else:
                    cur_note = temp
                    row[i] = can1
                    cnt += 1
            else:
                if cnt < 3 and abs(can1 - can2) < THRESHOLD:
                    cur_note = temp
                    row[i] = can1
                    cnt += 1
                else :
                    row[i] = can2
                    cnt = 1
        err_table.append(row)

    #print("backtracking")
    # backtracking
    ret = []


    cur_pos = data_len-1
    cur_note = predict_note-1
    
    this_note = [0, pitch_set[cur_pos][0]+FRAME_SIZE , 0] # onset offset pitch
    note_pitch = [pitch_set[cur_pos][1]]
    pitch_cnt = 1

    # O(N+M)
    while cur_pos > 0 and cur_note > 0:
        if err_table[cur_note][cur_pos] == err_table[cur_note-1][cur_pos-1]:
            this_note[0] = pitch_set[cur_pos][0]-FRAME_SIZE
            this_note[2] = np.median(np.array(note_pitch))
            # print(f'this_note is {this_note}')
            if this_note[2] < 35:
                continue
            ret.append(this_note)
            this_note = [0, pitch_set[cur_pos-1][0]+FRAME_SIZE, 0] # onset offset pitch
            note_pitch = [pitch_set[cur_pos-1][1]]
            cur_pos -= 1
            cur_note -= 1
            pitch_cnt = 1
        else :
            note_pitch.append(pitch_set[cur_pos-1][1])
            pitch_cnt += 1
            cur_pos -= 1
    return ret 

def process(pitch_set):
    res = []
    dat = trim_data(pitch_set)
    '''
    subset_cnt = 0
    '''

    for subset in dat:
        '''
        print(f'subset no: {subset_cnt}')
        subset_cnt += 1
        '''
        ans = solve(subset)    
        res += ans
    res.sort()
    toRemove = []
    for i in range(1,len(res)):
        if abs(res[i][2] - res[i-1][2]) <= 1 and abs(res[i][0] - res[i-1][1]) < 3*FRAME_SIZE:
            # print(res[i][0], res[i-1][0])
            res[i][0] = res[i-1][0]
            res[i][2] = int(round( (res[i][2] + res[i-1][2])/2 ))
            toRemove += [i-1]
    for idx in toRemove[::-1]:
        res.remove(res[idx])
    return res
    
