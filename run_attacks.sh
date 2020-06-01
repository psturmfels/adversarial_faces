#python run_adversarial_attacks.py --epsilon 0.2 --visible_devices 0 --attack_type community_naive_same
#python run_adversarial_attacks.py --epsilon 0.5 --visible_devices 0 --attack_type community_naive_same
#python run_adversarial_attacks.py --epsilon 0.7 --visible_devices 0 --attack_type community_naive_same
python postprocess.py --epsilon 0.2 --format png --attack_type community_naive_same
python postprocess.py --epsilon 0.5 --format png --attack_type community_naive_same
python postprocess.py --epsilon 0.7 --format png --attack_type community_naive_same



#python run_adversarial_attacks.py --epsilon 0.04 --visible_devices 0 --attack_type community_naive_same
#python run_adversarial_attacks.py --epsilon 0.04 --visible_devices 0 --attack_type community_naive_mean
#python run_adversarial_attacks.py --epsilon 0.02 --visible_devices 0 --attack_type community_naive_mean

