if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=180
model_name=DLinear
data_path=electricity/electricity.csv

python run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path $data_path \
  --model_id Electricity_$seq_len'_'96 \
  --model $model_name \
  --data other \
  --features S \
  --seq_len $seq_len \
  --label_len 30 \
  --pred_len 90 \
  --enc_in 128 \
  --des 'Exp' \
  --freq D \
  --itr 1 --batch_size 16  --learning_rate 0.01