# RLCoder_Extension

# set-up
The first step is to setting up your environment by installing the necessary dependencies listed in the `requirements.txt` file.
```bash
pip install -r requirements.txt
```


# training

python main.py --generator_model_path ~/Llama-2-7b-hf --enable_generation --inference_type unixcoder_with_rl --output_dir result/Llama+Contriever_normal --retriever_model_path ~/contriever-msmarco --data_per_epoch 5000


# evaluation

python main.py --eval --generator_model_path ~/Llama-2-7b-hf --enable_generation --inference_type unixcoder_with_rl --output_dir result/Llama+Contriever_normal --retriever_model_path ~/wsq/contriever-msmarco --data_per_epoch 5000

python main.py --eval --generator_model_path ~/Llama-2-7b-hf --enable_generation --debug --inference_type unixcoder_with_rl --output_dir result/Llama+Contriever_normal --retriever_model_path ~/wsq/contriever-msmarco --data_per_epoch 5000