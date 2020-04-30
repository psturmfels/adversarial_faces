#!/usr/bin/env bash

tmux new-session -d -s epsilon_2 '
python3 run_adversarial_attacks.py --visible_devices 0 --attack_type community_naive_random --epsilon 0.02 --model_path facenet_keras.h5;
read;
'

tmux new-session -d -s epsilon_4 '
python3 run_adversarial_attacks.py --visible_devices 1 --attack_type community_naive_random --epsilon 0.04 --model_path facenet_keras.h5;
read;
'

tmux new-session -d -s epsilon_6 '
python3 run_adversarial_attacks.py --visible_devices 2 --attack_type community_naive_random --epsilon 0.06 --model_path facenet_keras.h5;
read;
'

tmux new-session -d -s epsilon_8 '
python3 run_adversarial_attacks.py --visible_devices 3 --attack_type community_naive_random --epsilon 0.08 --model_path facenet_keras.h5;
read;
'

tmux new-session -d -s epsilon_10 '
python3 run_adversarial_attacks.py --visible_devices 4 --attack_type community_naive_random --epsilon 0.10 --model_path facenet_keras.h5;
read;
'
