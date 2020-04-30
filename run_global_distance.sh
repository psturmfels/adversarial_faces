#!/usr/bin/env bash

tmux new-session -d -s epsilon_2 '
python3 compute_distance_community.py --attack_type community_naive_random --epsilon 0.02;
read;
'

tmux new-session -d -s epsilon_4 '
python3 compute_distance_community.py --attack_type community_naive_random --epsilon 0.04;
read;
'

tmux new-session -d -s epsilon_6 '
python3 compute_distance_community.py --attack_type community_naive_random --epsilon 0.06;
read;
'

tmux new-session -d -s epsilon_8 '
python3 compute_distance_community.py --attack_type community_naive_random --epsilon 0.08;
read;
'

tmux new-session -d -s epsilon_10 '
python3 compute_distance_community.py --attack_type community_naive_random --epsilon 0.10;
read;
'
