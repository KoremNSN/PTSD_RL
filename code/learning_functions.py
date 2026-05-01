import pytensor.tensor as pt

def update_m1(left, right, chosen, rew, is_first, sub, Q_t, alpha):
    """
    Model 1 Update Function: Single Alpha
    """
    # Reset Q-values for the current subject if it's the first trial of a session
    Q_t_reset_row = pt.switch(is_first, pt.zeros(4), Q_t[sub])
    Q_t_reset = pt.set_subtensor(Q_t[sub], Q_t_reset_row)
    
    # Extract Q-values for presented options
    q_left = Q_t_reset[sub, left]
    q_right = Q_t_reset[sub, right]
    
    # Update chosen option
    PE = rew - Q_t_reset[sub, chosen]
    Q_t_next = pt.set_subtensor(Q_t_reset[sub, chosen], Q_t_reset[sub, chosen] + alpha[sub] * PE)
    
    return Q_t_next, q_left, q_right, Q_t_reset

def update_m2(left, right, chosen, rew, is_first, sub, Q_t, alpha_pos, alpha_neg):
    """
    Model 2 Update Function: Valence (Alpha split by PE > 0)
    """
    # Reset Q-values for the current subject if it's the first trial of a session
    Q_t_reset_row = pt.switch(is_first, pt.zeros(4), Q_t[sub])
    Q_t_reset = pt.set_subtensor(Q_t[sub], Q_t_reset_row)
    
    # Extract Q-values for presented options
    q_left = Q_t_reset[sub, left]
    q_right = Q_t_reset[sub, right]
    
    # Update chosen option
    PE = rew - Q_t_reset[sub, chosen]
    alpha_val = pt.switch(PE > 0, alpha_pos[sub], alpha_neg[sub])
    
    Q_t_next = pt.set_subtensor(Q_t_reset[sub, chosen], Q_t_reset[sub, chosen] + alpha_val * PE)
    
    return Q_t_next, q_left, q_right, Q_t_reset

def update_m3(left, right, chosen, rew, is_first, is_win_trial, sub, Q_t, alpha_win, alpha_lose):
    """
    Model 3 Update Function: Task (Alpha split by Win vs Lose trial domain)
    """
    # Reset Q-values for the current subject if it's the first trial of a session
    Q_t_reset_row = pt.switch(is_first, pt.zeros(4), Q_t[sub])
    Q_t_reset = pt.set_subtensor(Q_t[sub], Q_t_reset_row)
    
    # Extract Q-values for presented options
    q_left = Q_t_reset[sub, left]
    q_right = Q_t_reset[sub, right]
    
    # Update chosen option
    PE = rew - Q_t_reset[sub, chosen]
    alpha_val = pt.switch(is_win_trial, alpha_win[sub], alpha_lose[sub])
    
    Q_t_next = pt.set_subtensor(Q_t_reset[sub, chosen], Q_t_reset[sub, chosen] + alpha_val * PE)
    
    return Q_t_next, q_left, q_right, Q_t_reset
