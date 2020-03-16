#!/usr/bin/env bash

cd ..

tmux new-session -d -s epsilon_2 '
python3 run_compute_distance.py \
    --attack_type self_distance \
    --epsilon 0.02 \
    --output_file results/c_im_m_id_c_da/self_distance/epsilon_0.02_subsample_0.25.csv \
    --modify_identity \
    --subsample_rate 0.25;
python3 run_compute_distance.py \
    --attack_type self_distance \
    --epsilon 0.02 \
    --output_file results/c_im_m_id_c_da/self_distance/epsilon_0.02_subsample_0.5.csv \
    --modify_identity \
    --subsample_rate 0.5;
python3 run_compute_distance.py \
    --attack_type self_distance \
    --epsilon 0.02 \
    --output_file results/c_im_m_id_c_da/self_distance/epsilon_0.02_subsample_0.75.csv \
    --modify_identity \
    --subsample_rate 0.75;
read;
'

tmux new-session -d -s epsilon_4 '
python3 run_compute_distance.py \
    --attack_type self_distance \
    --epsilon 0.04 \
    --output_file results/c_im_m_id_c_da/self_distance/epsilon_0.04_subsample_0.25.csv \
    --modify_identity \
    --subsample_rate 0.25;
python3 run_compute_distance.py \
    --attack_type self_distance \
    --epsilon 0.04 \
    --output_file results/c_im_m_id_c_da/self_distance/epsilon_0.04_subsample_0.5.csv \
    --modify_identity \
    --subsample_rate 0.5;
python3 run_compute_distance.py \
    --attack_type self_distance \
    --epsilon 0.04 \
    --output_file results/c_im_m_id_c_da/self_distance/epsilon_0.04_subsample_0.75.csv \
    --modify_identity \
    --subsample_rate 0.75;
read;
'

tmux new-session -d -s epsilon_6 '
python3 run_compute_distance.py \
    --attack_type self_distance \
    --epsilon 0.06 \
    --output_file results/c_im_m_id_c_da/self_distance/epsilon_0.06_subsample_0.25.csv \
    --modify_identity \
    --subsample_rate 0.25;
python3 run_compute_distance.py \
    --attack_type self_distance \
    --epsilon 0.06 \
    --output_file results/c_im_m_id_c_da/self_distance/epsilon_0.06_subsample_0.5.csv \
    --modify_identity \
    --subsample_rate 0.5;
python3 run_compute_distance.py \
    --attack_type self_distance \
    --epsilon 0.06 \
    --output_file results/c_im_m_id_c_da/self_distance/epsilon_0.06_subsample_0.75.csv \
    --modify_identity \
    --subsample_rate 0.75;
read;
'

tmux new-session -d -s epsilon_8 '
python3 run_compute_distance.py \
    --attack_type self_distance \
    --epsilon 0.08 \
    --output_file results/c_im_m_id_c_da/self_distance/epsilon_0.08_subsample_0.25.csv \
    --modify_identity \
    --subsample_rate 0.25;
python3 run_compute_distance.py \
    --attack_type self_distance \
    --epsilon 0.08 \
    --output_file results/c_im_m_id_c_da/self_distance/epsilon_0.08_subsample_0.5.csv \
    --modify_identity \
    --subsample_rate 0.5;
python3 run_compute_distance.py \
    --attack_type self_distance \
    --epsilon 0.08 \
    --output_file results/c_im_m_id_c_da/self_distance/epsilon_0.08_subsample_0.75.csv \
    --modify_identity \
    --subsample_rate 0.75;
read;
'

tmux new-session -d -s epsilon_10 '
python3 run_compute_distance.py \
    --attack_type self_distance \
    --epsilon 0.1 \
    --output_file results/c_im_m_id_c_da/self_distance/epsilon_0.10_subsample_0.25.csv \
    --modify_identity \
    --subsample_rate 0.25;
python3 run_compute_distance.py \
    --attack_type self_distance \
    --epsilon 0.1 \
    --output_file results/c_im_m_id_c_da/self_distance/epsilon_0.10_subsample_0.5.csv \
    --modify_identity \
    --subsample_rate 0.5;
python3 run_compute_distance.py \
    --attack_type self_distance \
    --epsilon 0.1 \
    --output_file results/c_im_m_id_c_da/self_distance/epsilon_0.10_subsample_0.75.csv \
    --modify_identity \
    --subsample_rate 0.75;
read;
'
