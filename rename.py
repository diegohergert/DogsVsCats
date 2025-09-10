import os

def rename(path, prefix):
    "Renames images in path to prefix_index.jpg"
    for i, filename in enumerate(os.listdir(path)):
        if filename.endswith(".jpg"):
            new_name = f"{prefix}_{i}.jpg"
            os.rename(os.path.join(path, filename), os.path.join(path, new_name))


rename("CatImages\cats", "cat")
rename("DogImages\dogs", "dog")
print("Renaming complete.")