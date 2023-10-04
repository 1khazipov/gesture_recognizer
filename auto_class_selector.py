import json

# SELECT CLASSES TO EXTRACT!
classes_to_extract = ["before", "thin"]

class_file = open("dataset/wlasl_class_list.txt", "r")
class_data = class_file.read().split("\n")
class_file.close()

classes_indexes = []

for i in range(0, len(class_data)-1):
    line = class_data[i]
    pair = line.strip().split('\t')
    index = int(pair[0])
    value = pair[1]
    if value in classes_to_extract:
        classes_indexes.append(index)

videos = json.load(open("dataset/nslt_2000.json"))

new_data_file = {}

for key in videos:
    if int(videos[key]["action"][0]) in classes_indexes:
        new_data_file[key] = videos[key]

result_file = open("output.json", "w")
result_file.write(json.dumps(new_data_file))
result_file.close()