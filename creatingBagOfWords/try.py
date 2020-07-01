dictionary = {}

arr = ["x", "y", "x"]
arr2 = ["x", "y"]

for i in range(len(arr2)):
    cont = 0
    for j in range(len(arr)):
        if arr[j] == arr2[i]:
            cont += 1
    dictionary.update({arr2[i]: cont})

print(dictionary)
