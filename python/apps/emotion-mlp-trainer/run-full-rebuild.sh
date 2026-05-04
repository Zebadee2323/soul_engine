rm -rf .venv/
bash setup-tf-cuda-venv.sh
#bash sources/run-uldisvalainis-audio-emotions.sh
#bash sources/run-ejlok1-cremad.sh
bash sources/run-piyushagni5-berlin-database-of-emotional-speech-emodb.sh
bash run-dataset-transform.sh
bash run-train-emotion-mlp.sh
bash run-predict-emotion.sh ./test_data/
