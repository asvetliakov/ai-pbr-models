import json

with open("./matsynth_final_indexes.json", "r") as f:
    final_index_data = json.load(f)

with open("./matsynth_name_index_map.json", "r") as f:
    name_index_map = json.load(f)

with open("./matsynth_samples_check.json", "r") as f:
    all_matsynth_sample_data = json.load(f)


print(f"All indexes len: {len(final_index_data["all_valid_indexes"])}")
print(f"Category mapping len {len(final_index_data["new_category_mapping"])}")

# print("Checking final_index_data consistency...")
# for idx in final_index_data["all_valid_indexes"]:
#     name = name_index_map["index_to_name"].get(str(idx), None)
#     if name is None:
#         print(f"!!!Index {idx} not found in name_index_map, skipping.")
#         continue

#     if final_index_data["new_category_mapping"].get(name, None) is None:
#         print(
#             f"!!!Index {idx} with name {name} not found in new_category_mapping, skipping."
#         )
#         continue

#     if (
#         idx not in final_index_data["same_diffuse_albedo_indexes"]
#         and idx not in final_index_data["different_diffuse_albedo"]
#     ):
#         print(
#             f"!!!Index {idx} not found in same_diffuse_albedo or different_diffuse_albedo, skipping."
#         )
#         continue

# print("Checking new_category_mapping consistency...")
# if final_index_data.get("without_diffuse_category_mapping", None) is None:
#     final_index_data["without_diffuse_category_mapping"] = {}

# remove_keys = []

# for name, category in final_index_data["new_category_mapping"].items():
#     idx = name_index_map["name_to_index"].get(name, None)
#     if idx is None:
#         print(f"!!!Name {name} not found in name_index_map, skipping.")
#         continue

#     if idx not in final_index_data["all_valid_indexes"]:
#         print(
#             f"!!!Name {name} with index {idx} not found in all_valid_indexes, skipping."
#         )
#         if idx in final_index_data["without_diffuse"]:
#             print(f" - without_diffuse")

#         remove_keys.append(name)
#         # final_index_data["new_category_mapping"].pop(name, None)
#         final_index_data["without_diffuse_category_mapping"][name] = category

# for key in remove_keys:
#     final_index_data["new_category_mapping"].pop(key, None)

# # with open("./matsynth_final_indexes.json", "w") as f:
# #     json.dump(final_index_data, f, indent=4)
