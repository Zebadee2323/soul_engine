source .venv/bin/activate
rm -rf ./train_data/
python3 ./dataset-transform.py --source-data-dir ./source_data --output-dir ./train_data/
