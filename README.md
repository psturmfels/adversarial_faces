# Adversarial Faces
A repository for exploring how adversarial machine learning can provide privacy against face recognition. 

# Code Interface
* `preprocess_vggface.py` has two modes of operation:
  * both rely on the VGG Face dataset being under `image_directory` organized as `identity/image.jpg`
  * `preprocess` reads in the VGG Face-downloaded `bbox_file`, crops the images, prewhitens them, and saves them *all* under one giant `.h5` file with path `output_directory/nXXXXXX/images.h5`
  * `write_embeddings` generates embeddings from the model and the images written before and saves them under a giant `.h5` file with path `output_directory/nXXXXXX/embeddings.h5`
* `run_adversarial_attacks.py` generates adversarially perturbed images and saves them under `output_directory/nXXXXXX/attack_type/epsilon_XX.h5`. The resulting file has two datasets: `embeddings` and `images`
