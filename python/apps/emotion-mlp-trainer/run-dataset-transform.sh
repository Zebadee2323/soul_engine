source .venv/bin/activate
rm -rf ./train_data/
python3 ./dataset-transform.py --max-wavs-per-emotion 3000 --output-dir ./train_data/
