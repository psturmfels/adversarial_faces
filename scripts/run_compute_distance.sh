#!/usr/bin/env bash

cd ..

tmux new-session -d -s attack_none '
python3 run_compute_distance.py --attack_type none --epsilon 0.0;
read;
'

# tmux new-session -d -s epsilon_4 '
# python3 run_compute_distance.py --attack_type none --epsilon 0.04;
# read;
# '

# tmux new-session -d -s epsilon_6 '
# python3 run_compute_distance.py --attack_type none --epsilon 0.06;
# read;
# '

# tmux new-session -d -s epsilon_8 '
# python3 run_compute_distance.py --attack_type none --epsilon 0.08;
# read;
# '

# tmux new-session -d -s epsilon_10 '
# python3 run_compute_distance.py --attack_type none --epsilon 0.1;
# read;
# '
