#!/usr/bin/env bash

cd ..

tmux new-session -d -s epsilon_2 '
python3 run_compute_distance.py \
    --attack_type self_distance \
    --epsilon 0.02 \
    --output_file results/m_im_m_id_m_da/self_distance/epsilon_0.02.csv \
    --modify_identity \
    --modify_image \
    --modify_dataset;
'

tmux new-session -d -s epsilon_4 '
python3 run_compute_distance.py \
    --attack_type self_distance \
    --epsilon 0.04 \
    --output_file results/m_im_m_id_m_da/self_distance/epsilon_0.04.csv \
    --modify_identity \
    --modify_image \
    --modify_dataset;
'

tmux new-session -d -s epsilon_6 '
python3 run_compute_distance.py \
    --attack_type self_distance \
    --epsilon 0.06 \
    --output_file results/m_im_m_id_m_da/self_distance/epsilon_0.06.csv \
    --modify_identity \
    --modify_image \
    --modify_dataset;
'

tmux new-session -d -s epsilon_8 '
python3 run_compute_distance.py \
    --attack_type self_distance \
    --epsilon 0.08 \
    --output_file results/m_im_m_id_m_da/self_distance/epsilon_0.08.csv \
    --modify_identity \
    --modify_image \
    --modify_dataset;
'

tmux new-session -d -s epsilon_10 '
python3 run_compute_distance.py \
    --attack_type self_distance \
    --epsilon 0.1 \
    --output_file results/m_im_m_id_m_da/self_distance/epsilon_0.10.csv \
    --modify_identity \
    --modify_image \
    --modify_dataset;
'
