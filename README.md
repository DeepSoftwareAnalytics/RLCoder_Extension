# RLCoder_Extension

# training

test 
python main.py --generator_model_path ~/wsq/Llama-2-7b-hf --enable_generation --inference_type unixcoder_with_rl --output_dir result/Llama+Contriever_normal --retriever_model_path ~/wsq/contriever-msmarco --data_per_epoch 5000


# evaluation

python main.py --eval --generator_model_path ~/wsq/Llama-2-7b-hf --enable_generation --inference_type unixcoder_with_rl --output_dir result/Llama+Contriever_normal --retriever_model_path ~/wsq/contriever-msmarco --data_per_epoch 5000

python main.py --eval --generator_model_path ~/wsq/Llama-2-7b-hf --enable_generation --debug --inference_type unixcoder_with_rl --output_dir result/Llama+Contriever_normal --retriever_model_path ~/wsq/contriever-msmarco --data_per_epoch 5000