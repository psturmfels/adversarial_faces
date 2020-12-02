import os
import sys
import json
import glob
import time
import argparse
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person
from azure.cognitiveservices.vision.face.models import APIErrorException
from tqdm import tqdm
import numpy as np


def flags():
    parser = argparse.ArgumentParser(
        description="Command Line Interface to upload photos to Azure instance and create person group")
    parser.add_argument(
        '--image_directory',
        default="/Users/ie5/azureface/decoys",
        help='Top level directory for images',
        type=str
    )
    parser.add_argument(
        '--epsilon',
        default=0.1,
        help='Maximum perturbation distance for adversarial attacks.',
        type=float
    )
    parser.add_argument(
        '--attack_strategy',
        default='community_naive_mean',
        help='The attack strategy to use',
        type=str
    )
    parser.add_argument(
        '--auth_file',
        default='azure_auth.json',
        help='A json file containing the authenticaiton secret and endpoint for the Azure face service',
        type=str
    )
    parser.add_argument(
        '--num_clean',
        default=10,
        help='The number of clean photos to include in the database for each identity',
        type=int
    )
    parser.add_argument(
        '--num_decoys',
        default=5,
        help='The number of decoy photos from each identity that is provided for each other identity',
        type=int
    )
    return parser.parse_args()


class PersonGroupExistsError(ValueError):
    pass


class PersonGroupInterface:
    def __init__(self, person_group_name, endpoint, key):
        self.face_client = FaceClient(
            endpoint, CognitiveServicesCredentials(key))
        self.person_group_name = person_group_name
        self._verify_person_group_does_not_exist(person_group_name)
        self.name_to_person_obj = {}
        self.face_client.person_group.create(
            person_group_id=self.person_group_name,
            name=self.person_group_name
        )

    def _verify_person_group_does_not_exist(self, person_group_name):
        # Check if person group exists on Azure instance
        existing_pg = self.face_client.person_group.list()
        for gr in existing_pg:
            if gr.name == person_group_name or gr.person_group_id == person_group_name:
                raise PersonGroupExistsError(
                    f"Group {person_group_name} already exists; please delete or fix accordingly"
                )

    def create_person_for_each_identity(self, identities):
        for person_name in identities:
            self.name_to_person_obj[person_name] = self.face_client.person_group_person.create(
                self.person_group_name, person_name
            )

    def _add_folder(self, folder_path, person_name, indices):
        '''
        add all images at folder_path between alphabetically sorted from_indx and to_indx
        to person of person_name in the Azure group
        '''
        # List all jpeg, jpg and png images
        # glob returns full paths
        file_paths = [fn for fn in glob.glob(
            os.path.join(folder_path, '*')
        ) if fn.endswith("png") or fn.endswith("jpg") or fn.endswith("jpeg")]

        # Restrict to the first limit instances only in alphabetical order
        file_paths = np.take(sorted(file_paths), indices)

        print(f"Adding folder {folder_path}")
        # Add to Azure instance
        for img_path in tqdm(file_paths):
            try:
                self.face_client.person_group_person.add_face_from_stream(
                    self.person_group_name,
                    self.name_to_person_obj[person_name].person_id,
                    open(img_path, "r+b")
                )
            except APIErrorException as e:
                print(f"Exception {e} for image {img_path}")
            # Sleep to avoid triggering rate limiters
            time.sleep(10)

    def add_images_for_person(
            self, image_directory, attack_strategy, person_name, num_clean, num_decoys, epsilon):
        # Get all the protected identities.
        # Remember our folder structure is ground_truth_identity/attack_strategy/protected_identity/epsilon_X/png/*.png
        folders_wildcard = os.path.join(image_directory, person_name, attack_strategy, "*")
        protected_folders = glob.glob(folders_wildcard)

        if len(protected_folders) < 1:
            print(f"For folder {folders_wildcard} no protected folders")
            return

        # 1. Add clean images truly belonging to this identity
        # When epsilon = 0.0, we have clean images and it doesn't matter which identity is "being protected"
        # as they are all unmodified images but duplicated in each identity.
        clean_folder = os.path.join(protected_folders[0], "epsilon_0.0", "png")

        clean_indices = np.random.choice(50, size=num_clean, replace=False)
        # For now, hard code that we take only 1  clean image - the first one alphabetically.
        self._add_folder(clean_folder, person_name, clean_indices)

        # 2. Add decoy images belonging to this person_name in reality but modified to protect protected identities
        # For each identity, get one decoy at a different index.
        for indx, protected_identity_folder in enumerate(protected_folders):
            sampled_indices = np.random.choice(50, size=num_decoys, replace=False)

            full_folder_path = os.path.join(
                protected_identity_folder, f"epsilon_{epsilon}", "png")
            self._add_folder(full_folder_path, person_name, sampled_indices)


    def train(self):
        print()
        print('Training the person group...')
        # Train the person group
        self.face_client.person_group.train(self.person_group_name)

        while (True):
            training_status = self.face_client.person_group.get_training_status(self.person_group_name)
            print("Training status: {}.".format(training_status.status))
            print()
            if (training_status.status is TrainingStatusType.succeeded):
                break
            elif (training_status.status is TrainingStatusType.failed):
                sys.exit('Training the person group has failed.')
            time.sleep(5)


def main(argv=None):
    FLAGS = flags()

    with open(FLAGS.auth_file, "r") as f:
        auth_data = json.loads(f.read())

    epsilon_for_name = str(FLAGS.epsilon).replace(".", "p")
    person_group_name = f"{FLAGS.attack_strategy}_{FLAGS.num_clean}_{FLAGS.num_decoys}_{epsilon_for_name}"
    print(f"Building up person group {person_group_name}")

    pgi = PersonGroupInterface(
        person_group_name, auth_data["endpoint"], auth_data["key"])

    identities = [x for x in os.listdir(FLAGS.image_directory) if not x.startswith(".")]
    pgi.create_person_for_each_identity(identities)

    for identity in identities:
        pgi.add_images_for_person(
            FLAGS.image_directory,
            FLAGS.attack_strategy,
            identity,
            FLAGS.num_clean,
            FLAGS.num_decoys,
            FLAGS.epsilon
        )

    pgi.train()

if __name__ == '__main__':
    main()
