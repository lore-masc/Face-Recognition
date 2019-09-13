import json

example = {
    "training": [
        {"Training loss": 0.01579, "Training accuracy": 99.14},
        {"Test loss": 0.01600, "Test accuracy": 99.10},
        {"Training loss": 0.01894, "Training accuracy": 97.83},
        {"Test loss": 0.01021, "Test accuracy": 99.25},
        {"Training loss": 0.01008, "Training accuracy": 99.40},
        {"Test loss": 0.01021, "Test accuracy": 99.25}
    ],
    "prediction": {
        "files": [
            {
                "name": "10my-image.png",
                "probs": ['0.959', '0.015', '0.010', '0.010', '0.006'],
                "classes": ['Donato', 'Giulia', 'Angelo', 'Daniela', 'Damiano1']
            }
        ],
        "average": {'Angelo': 0.010161319747567177, 'Damiano1': 0.005751358345150948, 'Daniela': 0.00991909671574831,
                    'Donato': 0.9586997032165527, 'Giulia': 0.015468540601432323},
        "result": "Donato"
    }
}
print(json.dumps(example))
