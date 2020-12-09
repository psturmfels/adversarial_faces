import glob
import os
import time
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person
from azure.cognitiveservices.vision.face.models import APIErrorException

from tqdm import tqdm
import json
import numpy as np
import pandas as pd

auth_file = "/home/ivan/pascal_adversarial_faces/azure_auth.json"
with open(auth_file, "r") as f:
    auth_data = json.loads(f.read())
face_client = FaceClient(
    auth_data["endpoint"],
    CognitiveServicesCredentials(auth_data["key"])
)

def measure_azure_recall(
    face_client,
    azure_person_group_name,
    image_directory="/data/vggface/test_query_antisampled",
    num_query=10,
    verbose=False
):
    person_id_to_name = {}
    identities = []
    for p in face_client.person_group_person.list(person_group_id=azure_person_group_name):
        person_id_to_name[p.person_id] = p.name
        identities.append(p.name)

    discovery = []
    true = []
    identified_as = []


    for protector in tqdm(os.listdir(image_directory)):
        # We are sourcing query photos from epsilon_0.0.
        # In those cases, all subfolders in the "protected" identity have the same, clean
        # photo of the protector, so we just pick any single one that exists (e.g. n000958)
        # For the case where n000958 is itself the protector, n000958 is not present in its protected
        # subfolders, so we pick n000029 without loss of generality.
        if protector == "n000958":
            protected = "n000029"
        else:
            protected = "n000958"

        query_photos_paths = sorted(glob.glob(
            os.path.join(image_directory, protector, "*")
        ))

        for i in tqdm(np.random.choice(len(query_photos_paths), num_query)):
            chosen_path = query_photos_paths[i]

            # There should only be one face, so we use that as the query face.
            try:
                faces_in_query_photos = face_client.face.detect_with_stream(
                    open(chosen_path, "r+b"),
                    detectionModel='detection_02'
                )

                if len(faces_in_query_photos) != 1:
                    continue

                results = face_client.face.identify(
                    [faces_in_query_photos[0].face_id],
                    azure_person_group_name
                )

            except APIErrorException as e:
                print("API error exception", e, "for image", chosen_path)
                continue

            true.append(protector)

            if len(results) < 1 or len(results[0].candidates) < 1:
                discovery.append(0.0)
                identified_as.append("None")

            else:
                top_identity = person_id_to_name[results[0].candidates[0].person_id]

                identified_as.append(top_identity)

                # Note the switch of the term protector here:
                # protectors are also protected but we call them protectors because of the folder structure
                # In this case, the query photo belongs to the protector -- who is also protected by decoys
                # of *other* protectors. Therefore, if the identity returned is that of the "protector,"
                # this is a failure in the defense.
                if top_identity == protector:
                    discovery.append(1.0)
                else:
                    discovery.append(0.0)

            time.sleep(10)


    #if verbose:
    #    for true_id, recognized_id, query in zip(true, identified_as, paths_of_query):
    #        print("Face of {true_id} at {query} identitifed as {recognized_id}.".format(
    #            true_id=true_id, recognized_id=recognized_id, query=query, nearest=nearest))

    return np.mean(discovery)

results = []
for epsilon in [0.1, 0.25, 0.5]:
    for num_clean in [1, 5]:
        epsilon_for_name = str(epsilon).replace(".", "p")
        recall = measure_azure_recall(
            face_client,
            f"ensemble_casia-webface_{num_clean}_36_{epsilon_for_name}",
            num_query=5,
            verbose=True
        )
        results.append({
            "epsilon": epsilon,
            "num_clean": num_clean,
            "recall": recall
        })

results_df = pd.DataFrame(results).to_csv(
        f"/home/ivan/pascal_adversarial_faces/results/azure_recall_ensemble_casia-webface.csv")


#results = []
#for epsilon in [0.1, 0.25, 0.5]:
#    for num_clean in [1, 5]:
#        epsilon_for_name = str(epsilon).replace(".", "p")
#        recall = measure_azure_recall(
#            face_client,
#            f"mean_vggface2_{num_clean}_36_{epsilon_for_name}",
#            num_query=5,
#            verbose=True
#        )
#        results.append({
#            "epsilon": epsilon,
#            "num_clean": num_clean,
#            "recall": recall
#        })
#
#results_df = pd.DataFrame(results).to_csv(
#        f"/home/ivan/pascal_adversarial_faces/results/azure_recall_mean_vggface2.csv")
#recall = measure_azure_recall(
#        face_client,
#        "mean_vggface2_41_0_0p0",
#        num_query=5,
#        verbose=False
#        )
#print(f"Azure clean recall: {recall}")
#with open("/home/ivan/pascal_adversarial_faces/results/azure_recall_clean.txt", "a") as f:
#    f.write(f"{recall}\n")
