def get_possession_scores(agent):
    for (i, (state, action)) in enumerate(episode):
        possession_score = 0

        if state.ball_is_touched_by(agent.car_id):
            last_touch = i
            consecutive_touches = 1
            consecutive_touches_by_other_agent = 0
            opponent_touches = 0
        elif state.ball_is_touched:
            last_touch = i
            consecutive_touches = 0
            consecutive_touches_by_other_agent = 1
            opponent_touches = 1 if state.ball_is_touched_by_opponent(agent.car_id) else 0
        else:
            last_touch = None
            consecutive_touches = 0
            consecutive_touches_by_other_agent = 0
            opponent_touches = 0

        for (j, (future_state, future_action)) in enumerate(episode[i + 1:]):
            if future_state.ball_is_touched_by(agent.car_id):
                consecutive_touches += 1
                consecutive_touches_by_other_agent = 0
                last_touch = j
            elif future_state.ball_is_touched:
                last_touch = j
                consecutive_touches = 0
                consecutive_touches_by_other_agent = 1
                opponent_touches += 1

            if consecutive_touches > 0:
                possession_score += 1
            elif consecutive_touches_by_other_agent > 0:
                possession_score -= 1

            # another agent gained possession
            if consecutive_touches == 0 and consecutive_touches_by_other_agent >= 2:
                break

            # timeout for when ball isn't being influenced by any player
            if last_touch is not None and j - last_touch >= MAX_TICKS_SINCE_LAST_TOUCH:
                break

        yield possession_score / (len(trajectory) - i)