import glob
import os
import sys
import time
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person
from tqdm import tqdm
import json

class AzureInterface:
    def __init__(
        self, folder, endpoint, key, epsilon, attack_strategy, image_format, num_clean, include_decoys):

        self.face_client = FaceClient(
             endpoint, CognitiveServicesCredentials(key))

        epsilon_for_name = str(epsilon).replace(".", "p")
        self.person_group_name = f"{attack_strategy}_{num_clean}_{36 if include_decoys else 0}_{epsilon_for_name}"

        print(f"Building up person group {self.person_group_name}")

        for p in self.face_client.person_group.list():
            if p.name == self.person_group_name:
                print("Deleting existing group {self.person_group_name}")
                self.face_client.person_group.delete(person_group_id=self.person_group_name)

        self.face_client.person_group.create(
             person_group_id=self.person_group_name,
             name=self.person_group_name
         )

        self.log_file_path = os.path.join("azure_face_logfiles", f"{self.person_group_name}.txt")

        identities = os.listdir(folder)

        self.name_to_person_obj = {}

        for person_name in tqdm(identities):
            self.name_to_person_obj[person_name] = self.face_client.person_group_person.create(
                 self.person_group_name, person_name
             )

            attack_subfolder = os.path.join(folder, person_name, attack_strategy)

            protected = os.listdir(attack_subfolder)

            with open(self.log_file_path, "a") as f:
                f.write(f"{person_name}\n")

            if include_decoys:
                for indx, other_identity in enumerate(protected):
                    protected_folder = os.path.join(
                        attack_subfolder,
                        other_identity,
                        "epsilon_{eps}".format(eps=epsilon),
                        image_format
                    )

                    self._add_folder_for_person(
                        protected_folder,
                        person_name,
                        exclude_endings=None,
                        max_imgs=-1
                    )

            # These were the images we used for this person for adversarial modification
            used_images = [
                x.split("/")[-1] \
                for indx, x in enumerate(self.associated_paths) \
                if self.associated_identities[indx] == person_name
            ]

            clean_folder = os.path.join(
                folder, person_name, "community_naive_mean", protected[0], "epsilon_0.0", "png")
            self._add_folder_for_person(
                clean_folder,
                person_name,
                exclude_endings=set(used_images),
                max_imgs=num_clean
            )



    def _add_folder_for_person(self, folder, person_name, exclude_endings=None, max_imgs=-1):
        paths_list = glob.glob(os.path.join(folder, "*"))
        len_before = len(paths_list)
        if not (exclude_endings is None):
            paths_list = [x for x in paths_list if not (x.split("/")[-1] in exclude_endings)]

        if max_imgs > 0:
            paths_list = paths_list[:max_imgs]

        with open(self.log_file_path, "a") as f:
            for img_path in tqdm(paths_list):
                try:
                    self.face_client.person_group_person.add_face_from_stream(
                         self.person_group_name,
                         self.name_to_person_obj[person_name].person_id,
                         open(img_path, "r+b")
                    )
                    f.write(f"{img_path}\n")
                except APIErrorException as e:
                    print(f"Exception {e} for image {img_path}")
                # Sleep to avoid triggering rate limiters
                time.sleep(10)


    def train(self):
        print()
        print(f'Training the person group {self.person_group_name}')
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


if __name__ == "__main__":
    with open("azure_auth.json", "r") as f:
        auth_data = json.loads(f.read())

    for attack_name in ["mean_vggface2", "mean_casia-webface"]:
        for epsilon in [0.1, 0.25, 0.5]:
            for num_clean in [1, 5, 9]:
                ai = AzureInterface(
                        folder="/data/vggface/test_perturbed_sampled",
                        endpoint=auth_data["endpoint"],
                        key=auth_data["key"],
                        epsilon=epsilon,
                        attack_strategy=attack_name,
                        image_format="png",
                        num_clean=num_clean,
                        include_decoys=True
                    )

