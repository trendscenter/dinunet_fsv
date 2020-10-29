#!/usr/bin/python
from coinstac_pyprofiler import custom_profile_merger as pm

simulator_test_dir = './test'
json_folder = 'profiler_log'
pm.merge_computation_json(simulator_test_dir, 5, json_folder, "./test/profile_merge_results/", json_folder + "_test",
                          save_format="json|html")
