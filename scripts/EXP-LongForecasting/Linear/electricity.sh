# add --individual for DLinear-I
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DLinear
data_path=electricity/electricity.csv

python run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $data_path \
  --model_id Electricity_$seq_len'_'96 \
  --model $model_name \
  --data custom \
  --features S \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate 0.0001 # >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'96.log 

<<comment
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $data_path \
  --model_id Electricity_$seq_len'_'192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate 0.001 >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'192.log  

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $data_path \
  --model_id Electricity_$seq_len'_'336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 336 \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate 0.001  >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'336.log  

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $data_path \
  --model_id Electricity_$seq_len'_'720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len 720 \
  --enc_in 321 \
  --des 'Exp' \
  --itr 1 --batch_size 16  --learning_rate 0.001  >logs/LongForecasting/$model_name'_'electricity_$seq_len'_'720.log  
comment
