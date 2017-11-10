from firebase import firebase
import time, random


# IO
firebase = firebase.FirebaseApplication('https://dungeonsanddragons-6a9b1.firebaseio.com/', authentication=None)

# Write the result in firebase
while 1:
    result = firebase.get('/dice', None)
    if result:
        keys = list(result.keys())
        if keys is not None and len(keys) > 0:
            firebase.delete('/dice', keys[0])
    ndice = random.randrange(5)
    dice_dict = {'d20': [random.randrange(20)+1 for x in range(ndice)],
                            'd6': [random.randrange(20)+1 for x in range(ndice)]}
    print(dice_dict)
    firebase.post('/dice', dice_dict)
    time.sleep(5)