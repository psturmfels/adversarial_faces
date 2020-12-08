from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person
import time
import json

auth_file = "/home/ivan/pascal_adversarial_faces/azure_auth.json"
with open(auth_file, "r") as f:
    auth_data = json.loads(f.read())
face_client = FaceClient(
    auth_data["endpoint"],
    CognitiveServicesCredentials(auth_data["key"])
)

def train(person_group_name, face_client):
    print()
    print(f'Training the person group {person_group_name}')
    # Train the person group
    face_client.person_group.train(person_group_name)

    while (True):
        training_status = face_client.person_group.get_training_status(person_group_name)
        print("Training status: {}.".format(training_status.status))
        print()
        if (training_status.status is TrainingStatusType.succeeded):
            break
        elif (training_status.status is TrainingStatusType.failed):
            sys.exit('Training the person group has failed.')
        time.sleep(5)

train("mean_vggface2_1_36_0p1", face_client)
train("mean_vggface2_5_36_0p1", face_client)

