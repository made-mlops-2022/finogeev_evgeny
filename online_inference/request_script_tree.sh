#!/bin/bash
while :
do
	curl http://localhost:8000/health -X GET
	curl http://localhost:8000/predict -X POST -H "Content-Type: multipart/form-data" -F "file=@./test_data/X_test_tree.npy"
	sleep 1
done