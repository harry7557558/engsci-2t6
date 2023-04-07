import os
from PIL import Image, ExifTags


def load_image(filepath, convert):
    # https://stackoverflow.com/a/26928142
    image=Image.open(filepath)

    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation]=='Orientation':
            break

    try:
        exif = image._getexif()

        if exif[orientation] == 3:
            image=image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image=image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image=image.rotate(90, expand=True)

    except (AttributeError, TypeError, KeyError):
        pass

    return image.convert(convert)


images = sorted(os.listdir("raw/"))
os.system("rm train/*")

for i in range(0, len(images), 2):
    original, train = images[i:i+2]
    assert 'train' in train

    original = load_image("raw/"+original, "RGB")
    train = load_image("raw/"+train, "L")
    w, h = original.width, original.height
    assert (w, h) == (train.width, train.height)

    sc = min(1080 / max(w, h), 1.0)
    if sc != 1.0:
        w = round(sc*w)
        h = round(sc*h)
        original = original.resize((w, h))
        train = train.resize((w, h))

    filename = "{:02d}".format(i//2)
    print(filename, (w, h))
    original.save("train/"+filename+".jpg", quality=95)
    train.save("train/"+filename+".png")
