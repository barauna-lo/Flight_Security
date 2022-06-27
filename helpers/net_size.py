# alter net size in cfg file
def change_net_size(size, path):
    with open(path, 'r', encoding='utf-8') as file:
        data = file.readlines()
        i = 0
        for line in data:
            if line.startswith("width"):
                data[i] = f"width={size}\n"
            if line.startswith("height"):
                data[i] = f"height={size}\n"
            i += 1
    with open(path, 'w', encoding='utf-8') as file2:
        file2.writelines(data)
        file2.close()