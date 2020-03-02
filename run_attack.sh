#!/usr/bin/env bash

tmux new-session -d -s epsilon_2 '
python3 get_attack_performance.py --visible_devices 6 --attack_type self_distance --epsilon 0.02;
python3 get_attack_performance.py --visible_devices 6 --attack_type target_image --epsilon 0.02;
python3 get_attack_performance.py --visible_devices 6 --attack_type none --epsilon 0.02;
read;
'

tmux new-session -d -s epsilon_4 '
python3 get_attack_performance.py --visible_devices 2 --attack_type self_distance --epsilon 0.04;
python3 get_attack_performance.py --visible_devices 2 --attack_type target_image --epsilon 0.04;
python3 get_attack_performance.py --visible_devices 2 --attack_type none --epsilon 0.04;
read;
'

tmux new-session -d -s epsilon_6 '
python3 get_attack_performance.py --visible_devices 3 --attack_type self_distance --epsilon 0.06;
python3 get_attack_performance.py --visible_devices 3 --attack_type target_image --epsilon 0.06;
python3 get_attack_performance.py --visible_devices 3 --attack_type none --epsilon 0.06;
read;
'

tmux new-session -d -s epsilon_8 '
python3 get_attack_performance.py --visible_devices 4 --attack_type self_distance --epsilon 0.08;
python3 get_attack_performance.py --visible_devices 4 --attack_type target_image --epsilon 0.08;
python3 get_attack_performance.py --visible_devices 4 --attack_type none --epsilon 0.08;
read;
'

tmux new-session -d -s epsilon_10 '
python3 get_attack_performance.py --visible_devices 5 --attack_type self_distance --epsilon 0.1;
python3 get_attack_performance.py --visible_devices 5 --attack_type target_image --epsilon 0.1;
python3 get_attack_performance.py --visible_devices 5 --attack_type none --epsilon 0.1;
read;
'
