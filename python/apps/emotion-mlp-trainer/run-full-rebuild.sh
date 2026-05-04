rm -rf .venv/
bash setup-tf-cuda-venv.sh
bash sources/run-uldisvalainis-audio-emotions.sh
bash run-dataset-transform.sh
bash run-train-emotion-mlp.sh
