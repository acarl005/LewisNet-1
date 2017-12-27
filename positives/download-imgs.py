from chemspipy import ChemSpider 
from chemspipy.errors import ChemSpiPyServerError
import time

# read the API key from file and instantiate Chem Spider api client
with open("apikey", "r") as key:
    cs = ChemSpider(key.read().strip())

index = 0

# the "startfrom" file persists the ID of the molecule we last requested, check if that exists first
try:
    with open("startfrom", "r") as f:
        index = int(f.read().strip())
except:
    pass

while(True):
    try:
        compound = cs.get_compound(index)

        # if (compound.common_name):
            # print(compound.common_name)

        # save the image with the ID as the name
        with open("images/" + str(index) + ".png", "wb") as f:
            f.write(compound.image)
        print(".", end="", flush=True)

    except ChemSpiPyServerError as err:
        # skip over invalid IDs
        if ("Invalid ID" in err.args[0]):
            print("x", end="", flush=True)
        else:
            print("ERROR", index)
            print(err)
    index += 1
    # save progress to "startfrom" file
    with open("startfrom", "w") as f:
        f.write(str(index))
    # throttle so as to not overwhelm the API
    time.sleep(0.25)

